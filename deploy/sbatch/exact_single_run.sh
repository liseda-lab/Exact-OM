#!/bin/bash
#SBATCH --job-name=full_ncit2doid_local_small  # Job name
#SBATCH --output=exp/debug_new_approach/full_ncit2doid_local_small/slurm_job_%j.out       # Output file
#SBATCH --error=exp/debug_new_approach/full_ncit2doid_local_small/slurm_job_%j.err        # Error file
#SBATCH --partition=gpu_hi                        # Partition
#SBATCH --nodes=1                                # Number of nodes
#SBATCH --ntasks=1                               # Number of tasks
#SBATCH --cpus-per-task=16                      # Number of CPUs
#SBATCH --mem=60G                           # Memory
#SBATCH --time=0                         # Time limit
#SBATCH --gres=gpu:1                             # Number of GPUs

# Static Variables
# TAG=$(date +%s)
JOB_NAME=xact_single
EXP_DIR=exp/debug_new_approach/$JOB_NAME
CONFIG_FILE=$EXP_DIR/config.yaml
DATA_DIR=data/

# Config
DATA_DIR=data/ncit-doid
SOURCE=ncit.owl
TARGET=doid.owl
REFERENCE=train.tsv
FULL_REFERENCE=test.tsv
# CANDIDATES=test.cands.tsv
CANDIDATES=small.test.cands.tsv
# CANDIDATES=difficult_test_cands.tsv
MEMORY=60G

# Local Unsupervised
exact -s $DATA_DIR/$SOURCE -t $DATA_DIR/$TARGET -o $EXP_DIR -f $DATA_DIR/$FULL_REFERENCE -c $DATA_DIR/$CANDIDATES -y $CONFIG_FILE -l -e -m $MEMORY -d 0
# # Global Unsupervised
# exact -s $DATA_DIR/$SOURCE -t $DATA_DIR/$TARGET -o $EXP_DIR -f $DATA_DIR/$FULL_REFERENCE -y $CONFIG_FILE -l -e -m $MEMORY -d 0
