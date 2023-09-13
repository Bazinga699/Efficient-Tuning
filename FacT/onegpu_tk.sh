#!/usr/bin/env bash

DATASET=$1
LR=$2



WANDB_API_KEY=2a0ff77c64888b3bd539c7873069809fbfeb6059 WANDB_MODE=offline python /home/lijun07/code/Efficient-Tuning/FacT/fact_tk.py --dataset $DATASET --lr $LR




