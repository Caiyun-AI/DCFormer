lr=2e-5


pretrained_model=/home/pretrained_models/your_model_path
chinese_tokenizer_path=/home/your_tokenizer_path

dataset_dir=./data_dir
data_cache=./data_cache_dir_xm_1024
per_device_train_batch_size=1
per_device_eval_batch_size=1
gradient_accumulation_steps=1
output_dir=./output_dir

deepspeed_config_file=ds_zero2_no_offload.json

# deepspeed_config_file=ds_zero0.json


torchrun --nnodes 1 --nproc_per_node 2 run_clm_pt_dcformer.py \
    --deepspeed ${deepspeed_config_file} \
    --model_name_or_path ${pretrained_model} \
    --tokenizer_name_or_path ${chinese_tokenizer_path} \
    --dataset_dir ${dataset_dir} \
    --data_cache_dir ${data_cache} \
    --validation_split_percentage 0.1 \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size ${per_device_eval_batch_size} \
    --do_train \
    --do_eval \
    --seed 24 \
    --bf16 \
    --num_train_epochs 1 \
    --lr_scheduler_type cosine \
    --learning_rate ${lr} \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --logging_strategy steps \
    --logging_steps 1 \
    --save_strategy steps \
    --save_total_limit 1 \
    --save_steps 500 \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --preprocessing_num_workers 8 \
    --block_size 1024 \
    --output_dir ${output_dir} \
    --overwrite_output_dir \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --torch_dtype bfloat16 \
    --use_gradient_checkpointing True \
    --ddp_find_unused_parameters False \
    --remove_unused_columns True  \
    --q_chunk_size 128 \
    --small True \
    --compile True \
    --max_train_samples 40 \
    --max_eval_samples 20 


