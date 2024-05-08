#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH --gres=gpu:titanx:1
#SBATCH --array=1-30

# Declare output folder as variable
out_folder="results/cifar10/resnet110/sse"

export CUDNN_PATH=$HOME/.conda/envs/TF_KERAS_3_GPU/lib/python3.10/site-packages/nvidia/cudnn
export LD_LIBRARY_PATH=$CUDNN_PATH/lib:$HOME/TensorRT-8.6.1.6/lib:$LD_LIBRARY_PATH
export TF_ENABLE_ONEDNN_OPTS=0

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
    --validation_split=0.0 \
    --model_type="ResNet110v1" \
    --data_augmentation \
    --nesterov \
    --optimizer="sgd" \
    --use_case="cifar10" \
    --initial_lr=0.1 \
    --l2_reg=0.0003 \
    --lr_schedule="sse" \
    --checkpointing \
    --checkpoint_every=60 \
    --epochs=300
    #--bootstrapping
