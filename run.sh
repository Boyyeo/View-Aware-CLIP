CUDA_VISIBLE_DEVICES=1 python run_clip.py \
    --output_dir ./clip-roberta-finetuned \
    --model_name_or_path openai/clip-vit-large-patch14 \
    --data_dir $PWD/datasets/coco_data \
    --dataset_name ydshieh/coco_dataset_script \
    --dataset_config_name=2017 \
    --image_column image_path \
    --caption_column caption \
    --remove_unused_columns=False \
    --per_device_train_batch_size="1024" \
    --per_device_eval_batch_size="32" \
    --learning_rate="5e-5" --warmup_steps="0" --weight_decay 0.2 \
    --adam_beta2="0.98" --adam_beta1="0.9" --adam_epsilon="1e-6" \
    --overwrite_output_dir \
    --eval_steps 3 \
    --max_steps 10000  \
    --eval_accumulation_steps 100 \
    --fp16 \
    --do_eval  \
    --max_seq_length 30 \
    --freeze_vision_model \
    --view_data_csv_path 'datasets/control_view_train_have_back.csv' \
    --view_data_batch_size 12 \
    #--prompt_tuning \


