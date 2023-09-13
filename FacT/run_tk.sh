#!/usr/bin/env bash
LR=$1

rlaunch --gpu=1 --cpu=8 --memory=16000 --positive-tags=2080ti --max-wait-duration=24h --replica-max 1000000000 -- bash /home/lijun07/code/Efficient-Tuning/FacT/onegpu_tk.sh caltech101 $LR & \



rlaunch --gpu=1 --cpu=8 --memory=16000 --positive-tags=2080ti --max-wait-duration=24h --replica-max 1000000000 -- bash /home/lijun07/code/Efficient-Tuning/FacT/onegpu_tk.sh cifar $LR & \




rlaunch --gpu=1 --cpu=8 --memory=16000 --positive-tags=2080ti --max-wait-duration=24h --replica-max 1000000000 -- bash /home/lijun07/code/Efficient-Tuning/FacT/onegpu_tk.sh dtd $LR & \




rlaunch --gpu=1 --cpu=8 --memory=16000 --positive-tags=2080ti --max-wait-duration=24h --replica-max 1000000000 -- bash /home/lijun07/code/Efficient-Tuning/FacT/onegpu_tk.sh oxford_flowers102 $LR & \




rlaunch --gpu=1 --cpu=8 --memory=16000 --positive-tags=2080ti --max-wait-duration=24h --replica-max 1000000000 -- bash /home/lijun07/code/Efficient-Tuning/FacT/onegpu_tk.sh oxford_iiit_pet $LR & \




rlaunch --gpu=1 --cpu=8 --memory=16000 --positive-tags=2080ti --max-wait-duration=24h --replica-max 1000000000 -- bash /home/lijun07/code/Efficient-Tuning/FacT/onegpu_tk.sh svhn $LR & \




rlaunch --gpu=1 --cpu=8 --memory=16000 --positive-tags=2080ti --max-wait-duration=24h --replica-max 1000000000 -- bash /home/lijun07/code/Efficient-Tuning/FacT/onegpu_tk.sh sun397 $LR & \




rlaunch --gpu=1 --cpu=8 --memory=16000 --positive-tags=2080ti --max-wait-duration=24h --replica-max 1000000000 -- bash /home/lijun07/code/Efficient-Tuning/FacT/onegpu_tk.sh patch_camelyon $LR & \




rlaunch --gpu=1 --cpu=8 --memory=16000 --positive-tags=2080ti --max-wait-duration=24h --replica-max 1000000000 -- bash /home/lijun07/code/Efficient-Tuning/FacT/onegpu_tk.sh eurosat $LR & \




rlaunch --gpu=1 --cpu=8 --memory=16000 --positive-tags=2080ti --max-wait-duration=24h --replica-max 1000000000 -- bash /home/lijun07/code/Efficient-Tuning/FacT/onegpu_tk.sh resisc45 $LR & \




rlaunch --gpu=1 --cpu=8 --memory=16000 --positive-tags=2080ti --max-wait-duration=24h --replica-max 1000000000 -- bash /home/lijun07/code/Efficient-Tuning/FacT/onegpu_tk.sh diabetic_retinopathy $LR & \




rlaunch --gpu=1 --cpu=8 --memory=16000 --positive-tags=2080ti --max-wait-duration=24h --replica-max 1000000000 -- bash /home/lijun07/code/Efficient-Tuning/FacT/onegpu_tk.sh clevr_count $LR & \




rlaunch --gpu=1 --cpu=8 --memory=16000 --positive-tags=2080ti --max-wait-duration=24h --replica-max 1000000000 -- bash /home/lijun07/code/Efficient-Tuning/FacT/onegpu_tk.sh clevr_dist $LR & \




rlaunch --gpu=1 --cpu=8 --memory=16000 --positive-tags=2080ti --max-wait-duration=24h --replica-max 1000000000 -- bash /home/lijun07/code/Efficient-Tuning/FacT/onegpu_tk.sh dmlab $LR & \




rlaunch --gpu=1 --cpu=8 --memory=16000 --positive-tags=2080ti --max-wait-duration=24h --replica-max 1000000000 -- bash /home/lijun07/code/Efficient-Tuning/FacT/onegpu_tk.sh kitti $LR & \




rlaunch --gpu=1 --cpu=8 --memory=16000 --positive-tags=2080ti --max-wait-duration=24h --replica-max 1000000000 -- bash /home/lijun07/code/Efficient-Tuning/FacT/onegpu_tk.sh dsprites_loc $LR & \




rlaunch --gpu=1 --cpu=8 --memory=16000 --positive-tags=2080ti --max-wait-duration=24h --replica-max 1000000000 -- bash /home/lijun07/code/Efficient-Tuning/FacT/onegpu_tk.sh dsprites_ori $LR & \




rlaunch --gpu=1 --cpu=8 --memory=16000 --positive-tags=2080ti --max-wait-duration=24h --replica-max 1000000000 -- bash /home/lijun07/code/Efficient-Tuning/FacT/onegpu_tk.sh smallnorb_azi $LR & \





rlaunch --gpu=1 --cpu=8 --memory=16000 --positive-tags=2080ti --max-wait-duration=24h --replica-max 1000000000 -- bash /home/lijun07/code/Efficient-Tuning/FacT/onegpu_tk.sh smallnorb_ele $LR & \

