#!/usr/bin/env bash

DATASET=$1
ENUM=$2
MOED=$3
TAU=$4
SCHE=$5
t0=$6
tf=$7


WANDB_API_KEY=2a0ff77c64888b3bd539c7873069809fbfeb6059 WANDB_MODE=offline python /home/lijun07/code/Efficient-Tuning/FacT/fact_tt.py --dataset $DATASET --expert_num $ENUM --moe_dim $MOED --tau $TAU --temp_scheduler $SCHE --tau0 $t0 --tau_final $tf




