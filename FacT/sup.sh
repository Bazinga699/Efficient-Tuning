

rlaunch --gpu=1 --cpu=8 --memory=16000 --positive-tags=2080ti --max-wait-duration=24h --replica-max 1000000000 -- bash /home/lijun07/code/Efficient-Tuning/FacT/onegpu_tt.sh dsprites_loc  5 8 16  & \







rlaunch --gpu=1 --cpu=8 --memory=16000 --positive-tags=2080ti --max-wait-duration=24h --replica-max 1000000000 -- bash /home/lijun07/code/Efficient-Tuning/FacT/onegpu_tt.sh dsprites_loc  5 8 32  & \







rlaunch --gpu=1 --cpu=8 --memory=16000 --positive-tags=2080ti --max-wait-duration=24h --replica-max 1000000000 -- bash /home/lijun07/code/Efficient-Tuning/FacT/onegpu_tt.sh dsprites_ori  5 8 4  & \







rlaunch --gpu=1 --cpu=8 --memory=16000 --positive-tags=2080ti --max-wait-duration=24h --replica-max 1000000000 -- bash /home/lijun07/code/Efficient-Tuning/FacT/onegpu_tt.sh dsprites_ori  5 8 16  & \







rlaunch --gpu=1 --cpu=8 --memory=16000 --positive-tags=2080ti --max-wait-duration=24h --replica-max 1000000000 -- bash /home/lijun07/code/Efficient-Tuning/FacT/onegpu_tt.sh smallnorb_azi  5 8 2  & \




