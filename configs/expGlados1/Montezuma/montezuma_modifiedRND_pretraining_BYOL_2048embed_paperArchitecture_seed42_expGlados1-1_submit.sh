#!/bin/bash 
# --job-name=Montezuma_modifiedRND_pretraining_BYOL_2048embed_paperArchitecture_seed42_expGlados1-1_7day_batchjob

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

torchrun --nnodes 1 --nproc_per_node 1 --standalone main.py --train --num_env_per_process 64 --config_path=./configs/expGlados1/Montezuma/config_modifiedRND_pretraining_BYOL_2048embed_paperArchitecture_Montezuma.conf --log_name=montezuma_modifiedRND_pretraining_BYOL_2048embed_paperArchitecture_seed42_expGlados1-1 --save_model_path=checkpoints/expGlados1/Montezuma/montezuma_modifiedRND_pretraining_BYOL_2048embed_paperArchitecture_expGlados1-1.ckpt --seed=42 --gpu_id=0 --use_wandb --wandb_api_key=d012c9698bf568b1807b1cfe9ed56611311573e8
