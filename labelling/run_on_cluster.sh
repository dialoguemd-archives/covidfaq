#!/bin/bash
#SBATCH --partition=unkillable
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --time=0:20:00
#SBATCH --output=out.log
#SBATCH --error=err.log
##SBATCH --mail-type=all
##SBATCH --mail-user=$YOUR_EMAIL

module load python/3.7
module load cuda/10.0/cudnn/7.6

python label.py --mq-folder ../scrape/ --uq-csv ../data/website_questions_21032020_13h00.csv  --output out.txt --use-content
