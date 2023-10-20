#!/bin/bash

# In this example, we show how to train SimCSE using multiple GPU cards and PyTorch's distributed data parallel on supervised NLI dataset.
# Set how many GPUs to use

NUM_GPU=2

# Randomly set a port number
# If you encounter "address already used" error, just run again or manually set an available port id.
PORT_ID=$(expr $RANDOM + 1000)

# Allow multiple threads
export OMP_NUM_THREADS=2

# Use distributed data parallel
# /root/hdd/data/my-sup-simcse-test 
# If you only want to use one card, uncomment the following line and comment the line with "torch.distributed.launch"
# python train.py \  /root/data/ppo4cl/pro_instruct_2w_triple_1.csv  /root/hdd/data/SynCSE-partial-NLI.csv /root/hdd/sup-simcse-roberta-large
torchrun  --nproc_per_node $NUM_GPU --master_port $PORT_ID train.py \
    --model_name_or_path /root/data/test/sup-simcse-bert-large-uncased \
    --train_file /root/data/test/SynCSE-scratch-NLI.csv \
    --output_dir /root/hdd/data/my-sup-simcse-snli \
    --num_train_epochs 3 \
    --per_device_train_batch_size 128 \
    --learning_rate 2e-7 \
    --max_seq_length 32 \
    --evaluation_strategy steps \
    --metric_for_best_model stsb_spearman \
    --load_best_model_at_end \
    --eval_steps 50 \
    --save_steps 50 \
    --pooler_type cls_before_pooler \
    --overwrite_output_dir \
    --temp 0.05 \
    --do_train \
    --do_eval \
    --bf16 \
    "$@"
