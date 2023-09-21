#!/usr/bin/env bash
ENUM=$1
MOED=$2
TAU=$3
SCHE=$4
t0=$5
tf=$6

rlaunch --gpu=1 --cpu=8 --memory=16000 --positive-tags=2080ti --max-wait-duration=24h --replica-max 1000000000 -- bash /home/lijun07/code/Efficient-Tuning/FacT/onegpu_tt.sh caltech101  $ENUM $MOED $TAU $SCHE $t0 $tf   & \



rlaunch --gpu=1 --cpu=8 --memory=16000 --positive-tags=2080ti --max-wait-duration=24h --replica-max 1000000000 -- bash /home/lijun07/code/Efficient-Tuning/FacT/onegpu_tt.sh dtd  $ENUM $MOED $TAU $SCHE $t0 $tf   & \




rlaunch --gpu=1 --cpu=8 --memory=16000 --positive-tags=2080ti --max-wait-duration=24h --replica-max 1000000000 -- bash /home/lijun07/code/Efficient-Tuning/FacT/onegpu_tt.sh eurosat  $ENUM $MOED $TAU $SCHE $t0 $tf   & \


