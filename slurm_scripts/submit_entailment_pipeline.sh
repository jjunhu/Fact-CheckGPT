#!/bin/bash
#SBATCH -J entailment_pipeline_job           # Job name
#SBATCH -o logs_entailment/entailment_pipeline_job_%j.out    # Output file (%j expands to jobID)
#SBATCH -e logs_entailment/entailment_pipeline_job_%j.err    # Error file (%j expands to jobID)
#SBATCH --mail-type=ALL                      # Email notifications for all job states
#SBATCH --mail-user=jh2829@cornell.edu        # Email address for notifications
#SBATCH -N 1                                 # Number of nodes requested
#SBATCH --ntasks=1                           # Number of tasks (usually 1)
#SBATCH --cpus-per-task=4                    # Number of CPUs per task
#SBATCH --mem=96G                             # Memory per CPU
#SBATCH --gres=gpu:1                        # Number of GPUs
#SBATCH -t 168:00:00                          # Time limit 
#SBATCH --partition=nlplarge                    # Request partition

eval "$(conda shell.bash hook)"
conda activate nlp

# Run the entailment pipeline script
python run_entailment_pipeline.py $1 $2
