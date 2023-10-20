python eval_t5.py \
    --model_name_or_path /root/hdd/gense-base-plus \
    --pooler cls_before_pooler \
    --task_set sts \
    --mode test \
    --add_prompt \
    "$@"