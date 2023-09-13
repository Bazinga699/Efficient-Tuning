import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn import functional as F
from avalanche.evaluation.metrics.accuracy import Accuracy
from tqdm import tqdm
import numpy as np
import random
import timm
from timm.models import create_model
from timm.scheduler.cosine_lr import CosineLRScheduler
from argparse import ArgumentParser
from vtab import *
import yaml
import wandb
from timm.models.layers import trunc_normal_

D_type = {
    'caltech101':0,
    'cifar':0,
    'dtd':0,
    'oxford_flowers102':0,
    'oxford_iiit_pet':0,
    'sun397':0,
    'svhn':0,
    'diabetic_retinopathy':1,
    'eurosat':1,
    'patch_camelyon':1,
    'resisc45':1,
    'clevr_count':2,
    'clevr_dist':2,
    'dmlab':2,
    'dsprites_loc':2,
    'dsprites_ori':2,
    'kitti':2,
    'smallnorb_azi':2,
    'smallnorb_ele':2,
}

class MOEweight(nn.Module):
    def __init__(self, channel, expert_num, tau, hidden_dim = 8):
        super(MOEweight, self).__init__()
        # self.mlp = nn.Sequential(
        #     nn.Linear(channel, hidden_dim, bias=False),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(hidden_dim, expert_num, bias=False),
        # )
        # self.apply(self._init_weights)
        self.expert_num = expert_num
        self.tau = tau

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)

    def forward(self, x):
        y = torch.zeros((*x.mean(1).shape[:-1], self.expert_num)).type(x.dtype).to(x.device).fill_(1/self.expert_num)
        return y




def train(args, model, dl, opt, scheduler, epoch):
    train_matrics = {}
    test_matrics = {}
    print_freq=200
    model.train()
    model = model.cuda()
    pbar = tqdm(range(epoch))
    for ep in pbar:
        model.train()
        model = model.cuda()
        # pbar = tqdm(dl)
        for i, batch in enumerate(dl):
            x, y = batch[0].cuda(), batch[1].cuda()
            out = model(x)
            loss = F.cross_entropy(out, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if i % print_freq == 0:
                # print(f'batch {i}/{total_batch} loss: {loss} mem: {torch.cuda.max_memory_allocated()/(1024.0 * 1024.0)}MB')
                train_matrics['loss'] = loss
                train_matrics['lr'] = opt.param_groups[0]["lr"]
                wandb.log(train_matrics)
            
        if scheduler is not None:
            scheduler.step(ep)
        if ep % 10 == 9:
            acc = test(vit, test_dl)[1]
            if acc > args.best_acc:
                args.best_acc = acc
                save(args, model, acc, ep)
            pbar.set_description(str(acc) + '|' + str(args.best_acc))

            test_matrics['acc'] = acc
            test_matrics['best_acc'] = args.best_acc
            wandb.log(test_matrics)

    model = model.cpu()
    return model


@torch.no_grad()
def test(model, dl):
    model.eval()
    acc = Accuracy()
    # pbar = tqdm(dl)
    model = model.cuda()
    for batch in dl:  # pbar:
        x, y = batch[0].cuda(), batch[1].cuda()
        out = model(x).data
        acc.update(out.argmax(dim=1).view(-1), y, 1)

    return acc.result()


def fact_forward_attn(self, x):
    B, N, C = x.shape
    qkv = self.qkv(x)
    Factu_x = torch.bmm(
        x,
        (self.moe_weight(x) @ vit.FacTu.reshape(vit.expert_num, -1)).reshape(B, C, self.dim)
        )
    
    q = vit.FacTv(self.dp(self.q_FacTs(Factu_x)))
    k = vit.FacTv(self.dp(self.k_FacTs(Factu_x)))
    v = vit.FacTv(self.dp(self.v_FacTs(Factu_x)))

    qkv += torch.cat([q, k, v], dim=2) * self.s

    qkv = qkv.reshape(B, N, 3,
                      self.num_heads,
                      C // self.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]

    attn = (q @ k.transpose(-2, -1)) * self.scale
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)

    x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    proj = self.proj(x)
    proj += vit.FacTv(self.dp(self.proj_FacTs(
        torch.bmm(
        x,
        (self.moe_weight(x) @ vit.FacTu.reshape(vit.expert_num, -1)).reshape(B, C, self.dim)
        )
    ))) * self.s
    x = self.proj_drop(proj)
    return x


def fact_forward_mlp(self, x):
    B, N, C = x.shape
    h = self.fc1(x)  # B n 4c
    # print(x.size(), h.size())
    h += vit.FacTv(self.dp(self.fc1_FacTs(
        torch.bmm(
        x,
        (self.moe_weight(x) @ vit.FacTu.reshape(vit.expert_num, -1)).reshape(B, C, self.dim)
        )
        )).reshape(
        B, N, 4, self.dim)).reshape(
        B, N, 4 * C) * self.s
    x = self.act(h)
    x = self.drop(x)
    h = self.fc2(x)
    x = x.reshape(B, N, 4, C)
    h += vit.FacTv(self.dp(self.fc2_FacTs(
        torch.bmm(
        x.reshape(B, N*4, C),
        (self.moe_weight(x).mean(dim=1) @ vit.FacTu.reshape(vit.expert_num, -1)).reshape(B, C, -1)
        ).reshape(
        B, N, 4 * self.dim)))) * self.s
    x = self.drop(h)
    return x


def set_FacT(model, dim=8, s=1, expert_num=3, tau=1,moe_hidden=8):
    if type(model) == timm.models.vision_transformer.VisionTransformer:
        model.FacTu = nn.Parameter(torch.zeros(expert_num,768, dim))
        for matrix in model.FacTu:
            nn.init.xavier_uniform_(matrix)
        model.expert_num = expert_num
        model.FacTv = nn.Linear(dim, 768, bias=False)
        nn.init.zeros_(model.FacTv.weight)
    for _ in model.children():
        if type(_) == timm.models.vision_transformer.Attention:
            _.q_FacTs = nn.Linear(dim, dim, bias=False)
            _.k_FacTs = nn.Linear(dim, dim, bias=False)
            _.v_FacTs = nn.Linear(dim, dim, bias=False)
            _.moe_weight = MOEweight(768, expert_num=expert_num, tau=tau, hidden_dim=moe_hidden)
            _.proj_FacTs = nn.Linear(dim, dim, bias=False)
            _.dp = nn.Dropout(0.1)
            _.s = s
            _.dim = dim
            bound_method = fact_forward_attn.__get__(_, _.__class__)
            setattr(_, 'forward', bound_method)
        elif type(_) == timm.models.layers.mlp.Mlp:
            _.fc1_FacTs = nn.Linear(dim, dim * 4, bias=False)
            _.fc2_FacTs = nn.Linear(4 * dim, dim, bias=False)
            _.moe_weight = MOEweight(768, expert_num=expert_num, tau=tau, hidden_dim=moe_hidden)
            _.dim = dim
            _.s = s
            _.dp = nn.Dropout(0.1)
            bound_method = fact_forward_mlp.__get__(_, _.__class__)
            setattr(_, 'forward', bound_method)
        elif len(list(_.children())) != 0:
            set_FacT(_, dim, s, expert_num, tau, moe_hidden)


def set_seed(seed=0):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_config(dataset_name):
    with open('./configs/tt/%s.yaml' % (dataset_name), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


@torch.no_grad()
def save(args, model, acc, ep):
    model.eval()
    model = model.cpu()
    trainable = {}
    for n, p in vit.named_parameters():
        if 'FacT' in n or 'head' in n:
            trainable[n] = p.data
    torch.save(trainable, args.save_path +'/' + 'lastmodel.pt')
    # with open(args.save_path + '/' + args.dataset + '.log', 'w') as f:
    #     f.write(str(ep) + ' ' + str(acc))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dim', type=int, default=0)
    parser.add_argument('--scale', type=float, default=0)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=1e-4)
    parser.add_argument('--model', type=str, default='vit_base_patch16_224_in21k')
    parser.add_argument('--dataset', type=str, default='cifar')
    parser.add_argument('--save_path', type=str, default='/data/lora_output/FACT')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--expert_num', type=int, default=2)
    parser.add_argument('--moe_dim', type=int, default=8)
    parser.add_argument('--tau', type=int, default=1)
    parser.add_argument('--dataset_type', type=int, default=0)
    

    args = parser.parse_args()
    args.dataset_type = D_type[args.dataset]
    args.save_path = os.path.join(args.save_path, 'fact_tt_moe')
    args.save_path = os.path.join(args.save_path, args.dataset)
    name = args.dataset
    config = get_config(name)
    if args.dim == 0:
        args.dim = config['rank']
    if args.scale == 0:
        args.scale = config['scale']
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    args.lr *= args.expert_num
    wandb_dir = '/data/lora_output/0913'
    if not os.path.exists(wandb_dir):
        os.makedirs(wandb_dir)
    wandb.init(entity='bazinga699',project='fact', config=args, dir=wandb_dir, tags=[f'fact_tt_moe_avg_scalelr'], name=f'fact_tt-{args.dataset}')

    #args.lr = args.batch_size / 64 * args.lr
    print(args)
    seed = args.seed
    set_seed(seed)
    
    args.best_acc = 0
    vit = create_model(args.model, checkpoint_path='/data/pretrained_models/ViT-B_16.npz', drop_path_rate=0.1)
    train_dl, test_dl = get_data(name, batch_size=args.batch_size)
    
    set_FacT(vit, dim=args.dim, s=args.scale, expert_num=args.expert_num, moe_hidden=args.moe_dim, tau=args.tau)

    trainable = []
    vit.reset_classifier(get_classes_num(name))
    total_param = 0
    for n, p in vit.named_parameters():
        if 'FacT' in n or 'head' in n or 'moe' in n:
            trainable.append(p)
            if 'head' not in n:
                total_param += p.numel()
        else:
            p.requires_grad = False
    print('total_param', total_param/1000000,"M")
    opt = AdamW(trainable, lr=args.lr, weight_decay=args.wd)
    scheduler = CosineLRScheduler(opt, t_initial=100,
                                  warmup_t=10, lr_min=1e-5, warmup_lr_init=1e-6, decay_rate=0.1)
    vit = train(args, vit, train_dl, opt, scheduler, epoch=100)
    print('acc1:', args.best_acc)

