#!/bin/bash 
#
# You should only work under the /scratch/users/<username> directory.
#
# Example job submission script
#
# TODO:
#   - Set name of the job below changing "Test" value.
#   - Set the requested number of tasks (cpu cores) with --ntasks parameter.
#   - Select the partition (queue) you want to run the job in:
#     - short : For jobs that have maximum run time of 120 mins. Has higher priority.
#     - long  : For jobs that have maximum run time of 7 days. Lower priority than short.
#     - longer: For testing purposes, queue has 31 days limit but only 3 nodes.
#   - Set the required time limit for the job with --time parameter.
#     - Acceptable time formats include "minutes", "minutes:seconds", "hours:minutes:seconds", "days-hours", "days-hours:minutes" and "days-hours:minutes:seconds"
#   - Put this script and all the input file under the same directory.
#   - Set the required parameters, input and output file names below.
#   - If you do not want mail please remove the line that has --mail-type
#   - Put this script and all the input file under the same directory.
#   - Submit this file using:
#      sbatch examle_submit.sh

# -= Resources =-
#
#SBATCH --job-name=Montezuma_modifiedRND_BYOL_gradProj_2048embed_paperArchitecture_seed42_exp24-1_7day_batchjob
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=30
#SBATCH --partition=ai
#SBATCH --qos=ai
#SBATCH --account=ai
#SBATCH --gres=gpu:tesla_t4:1
#SBATCH --mem=32G
#SBATCH --time=168:00:00
#SBATCH --output=test-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=cgozpinar18@ku.edu.tr

################################################################################
##################### !!! DO NOT EDIT BELOW THIS LINE !!! ######################
################################################################################
# Load Anaconda
echo "======================="
echo "Loading Anaconda Module..."
#module load anaconda/2.7
module load anaconda/3.6
echo "======================="
. /kuacc/apps/anaconda/3.6/etc/profile.d/conda.sh
conda activate

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

torchrun --nnodes 1 --nproc_per_node 1 --rdzv-backend=c10d --rdzv-endpoint=127.0.0.1:0 --rdzv-id=241 main.py --train --num_env_per_process 64 --config_path=./configs/exp24/Montezuma/config_modifiedRND_BYOL_gradProj_2048embed_paperArchitecture_Montezuma.conf --log_name=montezuma_modifiedRND_BYOL_gradProj_2048embed_paperArchitecture_seed42_exp24-1 --save_model_path=checkpoints/exp24/Montezuma/montezuma_modifiedRND_BYOL_gradProj_2048embed_paperArchitecture_exp24-1.ckpt --seed=42 --use_wandb --wandb_api_key=d012c9698bf568b1807b1cfe9ed56611311573e8
