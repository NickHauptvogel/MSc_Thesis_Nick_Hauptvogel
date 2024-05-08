#!/bin/bash

#SBATCH --time=00:20:00
#SBATCH --gres=gpu:titanx:1
#SBATCH --array=1-10

# Declare output folder as variable
folder="imdb/"
out_folder="results/sse"

# If SLURM_ARRAY_TASK_ID is not set, set it to 1
if [ -z ${SLURM_ARRAY_TASK_ID+x} ]; then
    SLURM_ARRAY_TASK_ID=1
fi

nvidia-smi

# Run experiment
printf "\n\n* * * Run SGD for ID = $SLURM_ARRAY_TASK_ID. * * *\n\n\n"
python -m training \
    --id=$(printf "%02d" $SLURM_ARRAY_TASK_ID) \
    --seed=$SLURM_ARRAY_TASK_ID \
    --out_folder=$out_folder \
    --nesterov \
    --map_optimizer \
    --epochs=50 \
    --validation_split=0.2 \
    --checkpointing \
    --checkpoint_every=5 \
    --SSE_lr
    #--checkpoint_every_epoch \
    #--initial_lr=0.001
