#!/bin/bash

#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:titanx:1
#SBATCH --array=1-10

# Declare output folder as variable
out_folder="results/cifar100/resnet110/epoch_budget_300"

export CUDNN_PATH=$HOME/.conda/envs/TF_KERAS_3_GPU/lib/python3.10/site-packages/nvidia/cudnn
export LD_LIBRARY_PATH=$CUDNN_PATH/lib:$HOME/TensorRT-8.6.1.6/lib:$LD_LIBRARY_PATH
export TF_ENABLE_ONEDNN_OPTS=0

# If SLURM_ARRAY_TASK_ID is not set, set it to 1
if [ -z ${SLURM_ARRAY_TASK_ID+x} ]; then
    SLURM_ARRAY_TASK_ID=1
fi

# Run experiment
printf "\n\n* * * Run SGD for cluster size = $SLURM_ARRAY_TASK_ID. * * *\n\n\n"

# Compute budget as 300 / cluster size
budget=$((300 / $SLURM_ARRAY_TASK_ID))
echo "Budget: $budget"

# Train cluster size models in a for loop
for i in $(seq 1 $SLURM_ARRAY_TASK_ID)
do
    printf "\n\n* * * Run SGD for ID = ${SLURM_ARRAY_TASK_ID}_$i. * * *\n\n\n"

    python -m training \
        --out_folder=$out_folder \
        --validation_split=0.0 \
        --model_type="ResNet110v1" \
        --data_augmentation \
        --nesterov \
        --optimizer="sgd" \
        --use_case="cifar100" \
        --initial_lr=0.1 \
        --l2_reg=0.0003 \
        --lr_schedule="garipov" \
        --id=$(printf "%02d_%02d" $SLURM_ARRAY_TASK_ID $i) \
        --seed=$i \
        --epochs=$budget
done
