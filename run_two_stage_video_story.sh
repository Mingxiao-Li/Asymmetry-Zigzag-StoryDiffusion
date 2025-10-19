#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

python story_video_generation.py \
    --device cuda:0 \
    --image_model_path black-forest-labs/FLUX.1-dev \
    --save_dir ./test_result \
    --precision fp16 \
    --seed 46 \
    --window_length 10 \
    --benchmark_path None \
    --fix_seed 46 \
    --spsa_top_percent 0.2 \
    --concept_token girl \
    --subject_id_prompt "a 16 years old girl with long brown hair and blue eyes" \
    --prompt_list "wearing a green knitted hat" \
                  "dancing in a park" \
                  "playing with a dog" \
                  "reading a book in a library" \
                  "singing a song in a concert" \
                  "painting a picture in a art studio" \
                  "playing a guitar in a music festival" \
                  "riding a bike in a city park" \
                  "juggling balls in a street festival" \
                  "playing a game on a phone in a cafe"