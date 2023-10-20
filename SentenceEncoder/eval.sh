python evaluation.py \
    --model_name_or_path /root/hdd/mstsb-paraphrase-multilingual-mpnet-base-v2 \
    --pooler avg \
    --task_set sts \
    --mode test \
    "$@"