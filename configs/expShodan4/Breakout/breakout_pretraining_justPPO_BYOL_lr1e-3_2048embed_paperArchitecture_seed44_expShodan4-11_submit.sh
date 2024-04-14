#!/bin/bash 
# --job-name=Breakout_pretraining_justPPO_BYOL_lr1e-3_2048embed_paperArchitecture_seed44_expShodan4-11_7day_batchjob

################################################################################
##################### !!! DO NOT EDIT BELOW THIS LINE !!! ######################
################################################################################
# Load Anaconda
echo "======================="
echo "Activating conda env: rnd3"
conda activate rnd3
echo ""
echo "======================================================================================"


echo
echo "============================== ENVIRONMENT VARIABLES ==============================="
env
echo "===================================================================================="
echo
echo

# Set stack size to unlimited
echo "Setting stack size to unlimited..."
ulimit -s unlimited
ulimit -l unlimited
ulimit -a
echo

echo "==============================================================================="
# Command 1 for matrix
echo "Running Python script..."
# Put Python script command below
#python3 main.py --train --config_path=./configs/CartPole/config_rnd00.conf --log_name=CartPole_rnd00_5hr --save_model_path=checkpoints/CartPole/rnd00_5hr.ckpt

cd ../../../

torchrun --nnodes 1 --nproc_per_node 1 --rdzv-backend=c10d --rdzv-endpoint=127.0.0.1:0 --rdzv-id=411 main.py --train --num_env_per_process 64 --config_path=./configs/expShodan4/Breakout/config_justPPO_BYOL_lr1e-3_2048embed_paperArchitecture_Breakout.conf --log_name=breakout_pretraining_justPPO_BYOL_lr1e-3_2048embed_paperArchitecture_seed44_expShodan4-11 --save_model_path=checkpoints/expShodan4/Breakout/breakout_pretraining_justPPO_BYOL_lr1e-3_2048embed_paperArchitecture_expShodan4-11.ckpt --seed=44 --gpu_id=0 --use_wandb --wandb_api_key=d012c9698bf568b1807b1cfe9ed56611311573e8
