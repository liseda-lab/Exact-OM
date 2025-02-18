#!/bin/bash

# Static Variables
TAG=$(date +%s)
EXP_DIR=exp/${SLURM_JOB_NAME}
DATA_DIR=data/

# Config
SOURCE=ncit.owl
TARGET=snomed.owl
REFERENCE=train.tsv
CANDIDATES=test.cands.tsv


# Get the memory from the config file or default to 64G
MEMORY=$(grep 'max_heap:' $CONFIG_FILE | awk '{print $2}')
MEMORY=${MEMORY:-64G}

#SBATCH --job-name=matchadl$SOURCE2$TARGET_$TAG  # Job name
#SBATCH --output=$EXP_DIR/slurm_job_%j.out       # Output file
#SBATCH --error=$EXP_DIR/slurm_job_%j.err        # Error file
#SBATCH --partition=tier3                        # Partition
#SBATCH --nodelist=liseda-05                     # Node
#SBATCH --nodes=1                                # Number of nodes
#SBATCH --ntasks=1                               # Number of tasks
#SBATCH --cpus-per-task=4                        # Number of CPUs
#SBATCH --mem=$MEMORY                            # Memory
#SBATCH --time=02:00:00                          # Time limit
#SBATCH --gres=gpu:1                             # Number of GPUs

# Create experiment directory
mkdir -p $EXP_DIR

# Copy the config file to the experiment directory
mv /exp/config.yaml $EXP_DIR/config.yaml

# Run the Docker container
srun docker run --rm \
      -v $EXP_DIR:/code/exp \                               # Mount the experiment directory
      -v $DATA_DIR:/code/data \                             # Mount the data directory
      -w /code \                                            # Set the working directory
      --gpus all \                                          # Assign GPUs
      liseda-cluster.lasige.di.fc.ul.pt:5000/matchadl \     # Image
      /bin/bash -c "poetry run tensorboard --logdir=/code/exp/training_logs --host=0.0.0.0 & \
                        poetry run matchadl -s /code/data/$SOURCE -t /code/data/$TARGET -o /code/exp/ -r /code/data/$REFERENCE -c /code/data/$CANDIDATES -C /code/exp/config.yaml -l"
                        # Command to run in the container