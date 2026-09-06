#!/bin/bash
#SBATCH --job-name=sensitivity           # Job name
#SBATCH --output=out_%j.log  # Output log file (%j will be replaced with the job ID)
#SBATCH --error=error_%j.err # Error log file (%j will be replaced with the job ID)
#SBATCH --time=20-00:00:00                 # Maximum runtime
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --ntasks=1                       # Number of tasks (usually 1 for a single COMSOL instance)
#SBATCH --cpus-per-task=1               # Number of CPU cores per task
#SBATCH --mem=100GB                      # Memory per node (e.g., 32 GB)
 
# Define variables for the COMSOL run
python "./get_sensitivity_experiment_many_order.py" 
#python "./get_error_test.py" 

