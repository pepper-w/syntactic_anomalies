#!/bin/bash
DATA_DIR=DATA_PATH
TAG=RUN_TAG

python3 ./src/main.py \
    --train_tasks ModNoun_consis_size SubNObN VerbOb_consis_size AgreeShift_consis_size \
        SOMO bshift coord cola \
    --test_tasks ModNoun_consis_size SubNObN VerbOb_consis_size AgreeShift_consis_size \
    --data_dir ${DATA_DIR} >> ./logs/log_${TAG}.txt 2>&1