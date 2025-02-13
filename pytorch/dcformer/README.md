## pretrain script

bash run_pt.sh

## fine-tune script

bash run_sft.sh

## env

torch=2.5.1

deepspeed=0.16.0

transformers=4.46.3

accelerate=1.1.1



## Compare the time taken to run a single step between DCFormer-7B and Llama3-7B.

Compare the time taken to run a single step between DCFormer-7B and Llama3-7B when the context lengths are 1024 and 4096, respectively.

### when a sequence length is 1024
DCFormer (1024+not_compile) indicates that the text context length of DCFormer-7B's training data is 1024, and torch compilation is not used.

DCFormer(1024+compile) indicates that the text context length of DCFormer-7B's training data is 1024, and torch compilation is used.

Llama3(1024+flashattention+not_compile) indicates that the text context length of Llama3-7B's training data is 1024, FlashAttention is used and torch compilation is not used.

Llama3(1024+not_flashattention+not_compile) indicates that the text context length of Llama3-7B's training data is 1024, FlashAttention is not used and torch compilation is not used.

Llama3(1024+flashattention+compile) indicates that the text context length of Llama3-7B's training data is 1024, FlashAttention is used and torch compilation is used.

Llama3(1024+not_flashattention+compile) indicates that the text context length of Llama3-7B's training data is 1024, FlashAttention is not used and torch compilation is used.


This table presents the time taken, in seconds, for DCFormer-7B and Llama3-7B to execute a single step under various model configurations and settings, with a sequence length of 1024.

| model  | batch_size=18 | batch_size=20 | batch_size=24 | batch_size=26 | batch_size=28 | batch_size=30 |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| DCFormer(1024+not_compile)  | 34.117078  | 37.062407  | 44.023241  | 47.245348  | 50.583633  | OOM  |
| DCFormer(1024+compile)  | 8.85703  | 9.354807  | 10.424255  | 10.774785  | 11.618008  | 12.062406  |
| Llama3(1024+flashattention+not_compile)  | 9.034925  | OOM  | OOM  | OOM  | OOM  | OOM  |
| Llama3(1024+not_flashattention+not_compile)  | 9.084844  | OOM  | OOM  | OOM  | OOM  | OOM  |
| Llama3(1024+flashattention+compile)  | 8.354234  | 8.520253  | 8.727508  | 8.875685  | 9.15617  | 9.414761  |
| Llama3(1024+not_flashattention+compile)  | 10.627787  | 11.019346  | 11.677852  | 12.094193  | 12.400657  | 12.730574  |


### when a sequence length is 4096

DCFormer (4096+not_compile) indicates that the text context length of DCFormer-7B's training data is 4096, and torch compilation is not used.

DCFormer(4096+compile) indicates that the text context length of DCFormer-7B's training data is 4096, and torch compilation is used.

Llama3(4096+flashattention+not_compile) indicates that the text context length of Llama3-7B's training data is 4096, FlashAttention is used and torch compilation is not used.

Llama3(4096+not_flashattention+not_compile) indicates that the text context length of Llama3-7B's training data is 4096, FlashAttention is not used and torch compilation is not used.

Llama3(4096+flashattention+compile) indicates that the text context length of Llama3-7B's training data is 4096, FlashAttention is used and torch compilation is used.

Llama3(4096+not_flashattention+compile) indicates that the text context length of Llama3-7B's training data is 4096, FlashAttention is not used and torch compilation is used.


This table presents the time taken, in seconds, for DCFormer-7B and Llama3-7B to execute a single step under various model configurations and settings, with a sequence length of 4096.


| model  | batch_size=4 | batch_size=6 | batch_size=8 |
| ------------- | ------------- | ------------- | ------------- |
| DCFormer(4096+not_compile)  | 44.092943  | 62.446753  | OOM  |
| DCFormer(4096+compile)  | 10.559626  | 13.794642  | OOM  |
| Llama3(4096+flashattention+not_compile)  | 8.504642  | OOM  | OOM  |
| Llama3(4096+not_flashattention+not_compile)  | 8.557134  | OOM  | OOM  |
| Llama3(4096+flashattention+compile)  | 8.280823  | 8.908348  | 10.217685  |
| Llama3(4096+not_flashattention+compile)  | 11.566628  | 13.628004  | 15.812723  |
