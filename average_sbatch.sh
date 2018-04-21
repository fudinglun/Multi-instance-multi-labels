#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --mem=5GB
#SBATCH --job-name=Yelp
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=df1777@nyu.edu
#SBATCH --output=minmax_%j.out
#SBATCH --error=minmax_%j.err
#SBATCH --gres=gpu:1

module purge
module load pytorch/python3.6/0.3.0_4
source /scratch/df1777/yelp/py3.6.3/bin/activate

cd /scratch/df1777/yelp/dl_proj
python3 -u average_instance_model.py

deactivate