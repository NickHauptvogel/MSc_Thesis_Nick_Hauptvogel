#!/bin/bash

for i in {11..50}; do
    python -m training \
        --id=$(printf "%02d" $i) \
        --seed=$i \
        --out_folder=results/imdb_checkp \
        --nesterov \
        --map_optimizer \
        --epochs=5 \
        --validation_split=0.2 \
        --checkpointing \
        --checkpoint_every=1
done