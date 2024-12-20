CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --config_file configs/default_config.yaml distill.py \
    --teacher_generation_dataset_path /home/ksshumab/Alpaca/generated/alpaca_Qwen1.5-7B_finetune_2e-6_text2text.jsonl \
    --student_name Qwen/Qwen1.5-1.8B \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --num_train_epochs 3 \
    --output_dir /home/ksshumab/output_dir/Qwen1.5/alpaca_Qwen1.5-1.8B_ot_2e-6_text2text \
    --wandb_name "alpaca_Qwen1.5-1.8B_ot_2e-6_text2text" \
    --gradient_checkpointing \
    --eps 1e-8 \
    --learning_rate 2e-6 \
    --method forward_kl_text2text \
    --use_other_token yes \
    --use_norm linear \
    --norm_epsilon 1e-6 \
    --use_label_smoothing no \
    --smoothing_factor 0.1 \
    --student_temp 1 \
    --teacher_temp 1 \
