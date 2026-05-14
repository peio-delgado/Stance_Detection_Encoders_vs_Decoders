#!/bin/bash
#SBATCH --job-name=Stance_one
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1-00:00:00
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH --output=/home/pdelgado010/stance_project/stance_one.log
#SBATCH --error=/home/pdelgado010/stance_project/stance_one.err

cd /home/pdelgado010/stance_project
source /home/pdelgado010/envs/my_venv/bin/activate
export HF_HOME="/home/pdelgado010/.cache/huggingface"
export TRANSFORMERS_CACHE="/home/pdelgado010/.cache/huggingface"

export PYTHONUNBUFFERED=1

srun python stance_1shot.py