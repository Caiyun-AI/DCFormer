import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

tokenizer = AutoTokenizer.from_pretrained("Caiyun-AI/DCPythia-6.9B")
model = AutoModelForCausalLM.from_pretrained("Caiyun-AI/DCPythia-6.9B", trust_remote_code=True)

device = torch.device('cuda')
MAX_BATCH_SIZE = 1
MAX_SEQ_LENGTH = 2048
NUM_TOKENS_TO_GENERATE = 100
COMPILE = True

_ = model.to(device=device,dtype=torch.float16)
with torch.device(device):
    model.setup_caches(max_batch_size=MAX_BATCH_SIZE, max_seq_length=MAX_SEQ_LENGTH, set_kv_cache=True)

def decode_one_token(model, cur_token, input_pos):
    logits = model(cur_token, input_pos=input_pos, return_tensor=True)
    new_token = torch.argmax(logits[:, -1], dim=-1)[:,None]
    return new_token

prompt = "Beijing is the capital of China. London is the capital of"
input_ids = tokenizer.encode(prompt, return_tensors='pt')

compiled_decode_one_token = torch.compile(decode_one_token,mode="reduce-overhead", fullgraph=True) if COMPILE else None

with torch.no_grad():
    generated_ids = model.generate(input_ids.to(device),num_tokens_to_generate=NUM_TOKENS_TO_GENERATE, compiled_decode_one_token=compiled_decode_one_token)
    text = tokenizer.decode(generated_ids[0])
    print('generated text:', text)
