for DATASET in cifar caltech101 dtd oxford_flowers102 oxford_iiit_pet svhn sun397 patch_camelyon eurosat resisc45 diabetic_retinopathy clevr_count clevr_dist dmlab kitti dsprites_loc dsprites_ori smallnorb_azi smallnorb_ele
    do
        CUDA_VISIBLE_DEVICES=0 WANDB_API_KEY=2a0ff77c64888b3bd539c7873069809fbfeb6059 WANDB_MODE=offline  python fact_tt.py --dataset $DATASET
    done

for DATASET in cifar caltech101 dtd oxford_flowers102 oxford_iiit_pet svhn sun397 patch_camelyon eurosat resisc45 diabetic_retinopathy clevr_count clevr_dist dmlab kitti dsprites_loc dsprites_ori smallnorb_azi smallnorb_ele
    do
        CUDA_VISIBLE_DEVICES=0 WANDB_API_KEY=2a0ff77c64888b3bd539c7873069809fbfeb6059 WANDB_MODE=offline  python fact_tk.py --dataset $DATASET
    done
