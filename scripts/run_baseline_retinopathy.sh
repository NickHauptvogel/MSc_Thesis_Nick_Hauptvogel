#!/bin/bash

#SBATCH --time=3-00:00:00
#SBATCH --gres=gpu:titanx:1
#SBATCH --array=1-10

# Declare output folder as variable
out_folder="results/retinopathy/resnet50/bootstr"

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
    --batch_size=8 \
    --accumulation_steps=4 \
    --validation_split=0.0 \
    --epochs=90 \
    --model_type="ResNet50v1" \
    --initial_lr=0.00023072 \
    --l2_reg=0.00010674 \
    --checkpointing \
    --checkpoint_every=15 \
    --optimizer=sgd \
    --nesterov \
    --momentum=0.9901533 \
    --use_case="retinopathy" \
    --lr_schedule="retinopathy" \
    --image_size=512 \
    --bootstrapping
