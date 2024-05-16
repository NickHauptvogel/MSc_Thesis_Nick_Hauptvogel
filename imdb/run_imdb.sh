#!/bin/bash

#SBATCH --time=00:20:00
#SBATCH --gres=gpu:titanx:1
#SBATCH --array=1-10

####################################
# Declare variables
out_folder="results/imdb/bootstr"
epochs=50
checkpoint_every=5
options="--SSE_lr" # "" or "--SSE_lr" or "--bootstrapping"

####################################

# If SLURM_ARRAY_TASK_ID is not set, set it to 1
if [ -z ${SLURM_ARRAY_TASK_ID+x} ]; then
    SLURM_ARRAY_TASK_ID=1
fi

# Run experiment
printf "\n\n* * * Run SGD for ID = $SLURM_ARRAY_TASK_ID. * * *\n\n\n"
python -m training \
    --id=$(printf "%02d" $SLURM_ARRAY_TASK_ID) \
    --seed=$SLURM_ARRAY_TASK_ID \
    --out_folder=$out_folder \
    --nesterov \
    --map_optimizer \
    --epochs=$epochs \
    --validation_split=0.2 \
    --checkpointing \
    --checkpoint_every=$checkpoint_every \
    --SSE_lr
