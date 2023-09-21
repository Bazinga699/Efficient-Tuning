#!/usr/bin/env bash
ENUM=$1
MOED=$2
TAU=$3
SCHE=$4
t0=$5
tf=$6

rlaunch --gpu=1 --cpu=8 --memory=16000 --positive-tags=2080ti --max-wait-duration=24h --replica-max 1000000000 -- bash /home/lijun07/code/Efficient-Tuning/FacT/onegpu_tt.sh caltech101  $ENUM $MOED $TAU $SCHE $t0 $tf   & \



rlaunch --gpu=1 --cpu=8 --memory=16000 --positive-tags=2080ti --max-wait-duration=24h --replica-max 1000000000 -- bash /home/lijun07/code/Efficient-Tuning/FacT/onegpu_tt.sh cifar  $ENUM $MOED $TAU $SCHE $t0 $tf   & \




rlaunch --gpu=1 --cpu=8 --memory=16000 --positive-tags=2080ti --max-wait-duration=24h --replica-max 1000000000 -- bash /home/lijun07/code/Efficient-Tuning/FacT/onegpu_tt.sh dtd  $ENUM $MOED $TAU $SCHE $t0 $tf   & \




rlaunch --gpu=1 --cpu=8 --memory=16000 --positive-tags=2080ti --max-wait-duration=24h --replica-max 1000000000 -- bash /home/lijun07/code/Efficient-Tuning/FacT/onegpu_tt.sh oxford_flowers102  $ENUM $MOED $TAU $SCHE $t0 $tf   & \




rlaunch --gpu=1 --cpu=8 --memory=16000 --positive-tags=2080ti --max-wait-duration=24h --replica-max 1000000000 -- bash /home/lijun07/code/Efficient-Tuning/FacT/onegpu_tt.sh oxford_iiit_pet  $ENUM $MOED $TAU $SCHE $t0 $tf   & \




rlaunch --gpu=1 --cpu=8 --memory=16000 --positive-tags=2080ti --max-wait-duration=24h --replica-max 1000000000 -- bash /home/lijun07/code/Efficient-Tuning/FacT/onegpu_tt.sh svhn  $ENUM $MOED $TAU $SCHE $t0 $tf   & \




rlaunch --gpu=1 --cpu=8 --memory=16000 --positive-tags=2080ti --max-wait-duration=24h --replica-max 1000000000 -- bash /home/lijun07/code/Efficient-Tuning/FacT/onegpu_tt.sh sun397  $ENUM $MOED $TAU $SCHE $t0 $tf   & \




rlaunch --gpu=1 --cpu=8 --memory=16000 --positive-tags=2080ti --max-wait-duration=24h --replica-max 1000000000 -- bash /home/lijun07/code/Efficient-Tuning/FacT/onegpu_tt.sh patch_camelyon  $ENUM $MOED $TAU $SCHE $t0 $tf   & \




rlaunch --gpu=1 --cpu=8 --memory=16000 --positive-tags=2080ti --max-wait-duration=24h --replica-max 1000000000 -- bash /home/lijun07/code/Efficient-Tuning/FacT/onegpu_tt.sh eurosat  $ENUM $MOED $TAU $SCHE $t0 $tf   & \




rlaunch --gpu=1 --cpu=8 --memory=16000 --positive-tags=2080ti --max-wait-duration=24h --replica-max 1000000000 -- bash /home/lijun07/code/Efficient-Tuning/FacT/onegpu_tt.sh resisc45  $ENUM $MOED $TAU $SCHE $t0 $tf   & \




rlaunch --gpu=1 --cpu=8 --memory=16000 --positive-tags=2080ti --max-wait-duration=24h --replica-max 1000000000 -- bash /home/lijun07/code/Efficient-Tuning/FacT/onegpu_tt.sh diabetic_retinopathy  $ENUM $MOED $TAU $SCHE $t0 $tf   & \




rlaunch --gpu=1 --cpu=8 --memory=16000 --positive-tags=2080ti --max-wait-duration=24h --replica-max 1000000000 -- bash /home/lijun07/code/Efficient-Tuning/FacT/onegpu_tt.sh clevr_count  $ENUM $MOED $TAU $SCHE $t0 $tf   & \




rlaunch --gpu=1 --cpu=8 --memory=16000 --positive-tags=2080ti --max-wait-duration=24h --replica-max 1000000000 -- bash /home/lijun07/code/Efficient-Tuning/FacT/onegpu_tt.sh clevr_dist  $ENUM $MOED $TAU $SCHE $t0 $tf   & \




rlaunch --gpu=1 --cpu=8 --memory=16000 --positive-tags=2080ti --max-wait-duration=24h --replica-max 1000000000 -- bash /home/lijun07/code/Efficient-Tuning/FacT/onegpu_tt.sh dmlab  $ENUM $MOED $TAU $SCHE $t0 $tf   & \




rlaunch --gpu=1 --cpu=8 --memory=16000 --positive-tags=2080ti --max-wait-duration=24h --replica-max 1000000000 -- bash /home/lijun07/code/Efficient-Tuning/FacT/onegpu_tt.sh kitti  $ENUM $MOED $TAU $SCHE $t0 $tf   & \




rlaunch --gpu=1 --cpu=8 --memory=16000 --positive-tags=2080ti --max-wait-duration=24h --replica-max 1000000000 -- bash /home/lijun07/code/Efficient-Tuning/FacT/onegpu_tt.sh dsprites_loc  $ENUM $MOED $TAU $SCHE $t0 $tf   & \




rlaunch --gpu=1 --cpu=8 --memory=16000 --positive-tags=2080ti --max-wait-duration=24h --replica-max 1000000000 -- bash /home/lijun07/code/Efficient-Tuning/FacT/onegpu_tt.sh dsprites_ori  $ENUM $MOED $TAU $SCHE $t0 $tf   & \




rlaunch --gpu=1 --cpu=8 --memory=16000 --positive-tags=2080ti --max-wait-duration=24h --replica-max 1000000000 -- bash /home/lijun07/code/Efficient-Tuning/FacT/onegpu_tt.sh smallnorb_azi  $ENUM $MOED $TAU $SCHE $t0 $tf   & \





rlaunch --gpu=1 --cpu=8 --memory=16000 --positive-tags=2080ti --max-wait-duration=24h --replica-max 1000000000 -- bash /home/lijun07/code/Efficient-Tuning/FacT/onegpu_tt.sh smallnorb_ele  $ENUM $MOED $TAU $SCHE $t0 $tf   & \

