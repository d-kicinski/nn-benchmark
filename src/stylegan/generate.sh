#!/usr/bin/env bash

for size in 128 256 512 1024 ; do

checkpoint_path="/media/storage/models/stylegan-pytorch/stylegan-1024px-new.model"

time python generate.py \
    --size ${size} \
    --n_row 10 \
    --n_col 10 \
    --device "cpu" \
    --seed 0 \
    "${checkpoint_path}"
done


