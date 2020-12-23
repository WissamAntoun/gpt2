#!/usr/bin/env bash

export PYTHONPATH=../

python train_tpu.py \
 --config_file=../configs/mega.json \
 --input_file=gs://arabert_data/ar_gpt2/pretraining_data/* \
 --output_dir=gs://arabert_data/ar_gpt2/xlarge_pretraining_model \
 --max_seq_length=1024 \
 --do_train=True \
 --do_eval=False \
 --train_batch_size=256 \
 --eval_batch_size=256 \
 --learning_rate=1e-4 \
 --num_train_steps=1000000 \
 --num_warmup_steps=10000 \
 --save_checkpoints_steps=5000 \
 --iterations_per_loop=5000 \
 --use_tpu=True \
 --tpu_name="arabert128" \
 --num_tpu_cores=128
