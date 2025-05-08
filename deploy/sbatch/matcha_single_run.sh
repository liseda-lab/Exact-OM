#!/bin/bash
#SBATCH --job-name=1hop_subclass_of_all  # Job name
#SBATCH --output=exp/cluster/unsup_local_all_datasets/ncit2doid/1hop_subclass_of_all/slurm_job_%j.out       # Output file
#SBATCH --error=exp/cluster/unsup_local_all_datasets/ncit2doid/1hop_subclass_of_all/slurm_job_%j.err        # Error file
#SBATCH --partition=tier3                        # Partition
#SBATCH --nodelist=liseda-01                     # Node
#SBATCH --nodes=1                                # Number of nodes
#SBATCH --ntasks=1                               # Number of tasks
#SBATCH --cpus-per-task=15                      # Number of CPUs
#SBATCH --mem=60G                           # Memory
#SBATCH --time=02:00:00                          # Time limit
#SBATCH --gres=gpu:1                             # Number of GPUs

# Static Variables
# TAG=$(date +%s)
JOB_NAME=1hop_subclass_of_all
EXP_DIR=exp/cluster/unsup_local_all_datasets/ncit2doid/1hop_subclass_of_all
CONFIG_FILE=$EXP_DIR/config.yaml
DATA_DIR=data/

# Config
DATA_DIR=data/ncit-doid
SOURCE=ncit.owl
TARGET=doid.owl
REFERENCE=train.tsv
FULL_REFERENCE=test.tsv
CANDIDATES=test.cands.tsv

# Run the Docker container
# Local Supervised
# srun poetry run matchadl -s $DATA_DIR/$SOURCE -t $DATA_DIR/$TARGET -o $EXP_DIR -r $DATA_DIR/$REFERENCE -f $DATA_DIR/$FULL_REFERENCE -c $DATA_DIR/$CANDIDATES -y $CONFIG_FILE -l -e -m $MEMORY -d 0
# Local Unsupervised
matchadl -s $DATA_DIR/$SOURCE -t $DATA_DIR/$TARGET -o $EXP_DIR -f $DATA_DIR/$FULL_REFERENCE -c $DATA_DIR/$CANDIDATES -y $CONFIG_FILE -l -e -m 60G -d 0
# # Global Supervised
# srun poetry run matchadl -s $DATA_DIR/$SOURCE -t $DATA_DIR/$TARGET -o $EXP_DIR -r $DATA_DIR/$REFERENCE -f $DATA_DIR/$FULL_REFERENCE -y $CONFIG_FILE -l -e -m $MEMORY -d 0
# # Global Unsupervised
# srun poetry run matchadl -s $DATA_DIR/$SOURCE -t $DATA_DIR/$TARGET -o $EXP_DIR -f $DATA_DIR/$FULL_REFERENCE -y $CONFIG_FILE -l -e -m $MEMORY -d 0