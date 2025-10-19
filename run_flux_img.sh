#!/bin/bash
export CUDA_VISIBLE_DEVICES=6

python story_image_generation.py \
    --device cuda:0 \
    --model_path black-forest-labs/FLUX.1-dev \
    --save_dir ./test_result \
    --precision fp16 \
    --seed 42 \
    --window_length 10 \
    --benchmark_path None \
    --fix_seed 42 \
    --spsa_top_percent 0.2 \