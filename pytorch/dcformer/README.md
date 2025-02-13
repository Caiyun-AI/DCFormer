## pretrain script

bash run_pt.sh

## fine-tune script

bash run_sft.sh

## env

torch=2.5.1

deepspeed=0.16.0

transformers=4.46.3

accelerate=1.1.1



## dcformer7b和llama3-7b运行单个step测试速度对比

<img src="https://github.com/Caiyun-AI/DCFormer/blob/lbb/pytorch/dcformer/img/dcformer%E5%92%8Cllama3%E5%8D%95%E6%AD%A5%E8%BF%90%E8%A1%8C%E6%97%B6%E9%97%B4%E5%AF%B9%E6%AF%94.png" width="2000">


| model  | batch_size=18 | batch_size=20 | batch_size=24 | batch_size=26 | batch_size=28 | batch_size=30 |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| DCFormer(1024+not_compile)  | 34.117078  | 37.062407  | 44.023241  | 47.245348  | 50.583633  | OOM  |
| DCFormer(1024+compile)  | 8.85703  | 9.354807  | 10.424255  | 10.774785  | 11.618008  | 12.062406  |
| Llama3(1024+flashattention+not_compile)  | 9.034925  | OOM  | OOM  | OOMl  | OOM  | OOM  |
| Llama3(1024+not_flashattention+not_compile)  | 9.084844  | OOM  | OOM  | OOM  | OOM  | OOM  |
| Llama3(1024+flashattention+compile)  | 8.354234  | 8.520253  | 8.727508  | 8.875685  | 9.15617  | 9.414761  |
| Llama3(1024+not_flashattention+compile)  | 10.627787  | 11.019346  | 11.677852  | 12.094193  | 12.400657  | 12.730574  |
