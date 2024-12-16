lr=1e-4

pretrained_model=/home/pretrained_models/your_model_path
chinese_tokenizer_path=/home/your_tokenizer_path
dataset_dir=./dataset_dir
per_device_train_batch_size=1
per_device_eval_batch_size=1
gradient_accumulation_steps=1
output_dir=./output_dir

validation_file=dataset_dir/CoT_data.json

# deepspeed_config_file=ds_zero0.json
deepspeed_config_file=ds_zero2_no_offload.json

torchrun --nnodes 1 --nproc_per_node 2 run_clm_sft_with_peft.py \
    --deepspeed ${deepspeed_config_file} \
    --model_name_or_path ${pretrained_model} \
    --tokenizer_name_or_path ${chinese_tokenizer_path} \
    --dataset_dir ${dataset_dir} \
    --validation_split_percentage 0.001 \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size ${per_device_eval_batch_size} \
    --do_train \
    --do_eval  \
    --seed $RANDOM \
    --bf16 \
    --num_train_epochs 1 \
    --lr_scheduler_type cosine \
    --learning_rate ${lr} \
    --warmup_ratio 0.03 \
    --weight_decay 0 \
    --logging_strategy steps \
    --logging_steps 1 \
    --save_strategy steps \
    --save_total_limit 1 \
    --evaluation_strategy steps \
    --eval_steps 100 \
    --validation_file ${validation_file} \
    --save_steps 50 \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --preprocessing_num_workers 8 \
    --max_seq_length 1024 \
    --output_dir ${output_dir} \
    --overwrite_output_dir \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --torch_dtype bfloat16 \
    --use_gradient_checkpointing  True \
    --ddp_find_unused_parameters False \
    --remove_unused_columns True  \
    --q_chunk_size 128 \
    --small True \
    --compile True 
