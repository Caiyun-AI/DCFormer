import json
import os
from collections import defaultdict

os.environ["JAX_PLATFORMS"] = "cpu"

from smart_open import open
from flax.traverse_util import flatten_dict, unflatten_dict
import orbax.checkpoint
import orbax
import jax
import torch
import numpy as np
import argparse

def load_model(read_dir, load_step=None):
    step_prefix = "checkpoint"
    step_format_fixed_length = 8

    options = orbax.checkpoint.CheckpointManagerOptions()
    key = "default" # 'state' # model_path under step directory 
    item = {
        key: orbax.checkpoint.Checkpointer(orbax.checkpoint.PyTreeCheckpointHandler(use_ocdbt=True))
    }
    print('read_dir here', read_dir)
    mngr = orbax.checkpoint.CheckpointManager(read_dir, item, options)
    
    if load_step is None:
        load_step = mngr.latest_step()
        checkpoint_name = 'latest'
    else:
        checkpoint_name = f"{step_prefix}_" + str(load_step).zfill(step_format_fixed_length)

    print('load_step', load_step)
    with jax.default_device(jax.devices("cpu")[0]):
        weights = mngr.restore(load_step, items=item)
    
    # save torch model
    params = weights[key]['params']
    flat_weights = {".".join(k): v.astype('float16') for k, v in flatten_dict(params).items()}
    del weights
    for k, v in flat_weights.items():
        print(k, v.shape)
    print('load weights done')
    return flat_weights

def update_weight_from_maxtext(model, w, vocab_size=50304, num_blocks=2, model_dim=1024, num_heads=16, head_dim=128):
    map_dict={'w1': 'wi_0', 'w3': 'wi_1', 'w2': 'wo', 'weight': 'w', 'dd': 'dd.kernel', 'dw1': 'dw1.kernel'} # 'pre_proj': 'pre_proj', 'post_proj': 'post_proj'
    N, E, H, D = vocab_size, model_dim, num_heads, head_dim
    state_dict = {}
    for k, v in model.named_parameters():
        if k == 'tok_embeddings.weight':
            v = w['token_embedder.embedding'][:vocab_size,:]
        elif k == 'norm.weight':
            v = w['decoder.decoder_norm.scale']
        elif k == 'output.weight':
            v = w['decoder.logits_dense.kernel'].T[:vocab_size,:]  # E,N -> N,E
        else:
            layer = int(k.split('.')[1])
            sub_layer, _layer = layer % num_blocks, layer //num_blocks # sub_layer 0/1, _layer 0-12
            if '.attention.' in k:
                if k.endswith('_m') or 'dyn_w_proj.sw' in k:continue # merged proj weights
                if 'pre_proj.w' in k or 'post_proj.w' in k:
                    _, _, _, _, ptype, wtype = k.split('.') # dyn_w_proj
                else:
                    _, _, _, ptype, wtype = k.split('.')
                if k.endswith('_p'): continue # ablation parameters
                if ptype in ['dyn_w_proj', 'pre_proj', 'post_proj']: # pre post proj ; dw1, dd, qkw
                    v = w[f'decoder.layers.self_attention_{sub_layer}.AttentionOp_0.{map_dict.get(ptype, ptype)}.{map_dict.get(wtype, wtype)}'][:, _layer]
                elif ptype in ['q_norm', 'k_norm']:
                    v = w[f'decoder.layers.self_attention_{sub_layer}.{map_dict.get(ptype, ptype)}.scale'][:, _layer]
                elif ptype == 'wqkv':
                    _q = torch.tensor(w[f'decoder.layers.self_attention_{sub_layer}.query.kernel'][:, _layer]).reshape(E,H*D) # EHD->EE
                    _k = torch.tensor(w[f'decoder.layers.self_attention_{sub_layer}.key.kernel'][:, _layer]).reshape(E,H*D) # EHD->EE
                    _v = torch.tensor(w[f'decoder.layers.self_attention_{sub_layer}.value.kernel'][:, _layer]).reshape(E,H*D) # EHD->EE
                    v = torch.cat([_q, _k, _v],dim=-1).T
                else: # o
                    v = w[f'decoder.layers.self_attention_{sub_layer}.out.kernel'][:, _layer].reshape(H*D, E).T # HDE->E(HD)
            elif 'feed_forward' in k:
                ptype = k.split('.')[3] # w1, w3,w2,mgate_layer
                v = w[f'decoder.layers.mlp_{sub_layer}.{map_dict[ptype]}.kernel'][:,_layer].T
            elif 'ffn_norm' in k: # mlp layernorm
                v = w[f'decoder.layers.post_self_attention_layer_norm_{sub_layer}.scale'][:,_layer]
            elif 'attention_norm' in k: # attention layernorm
                v = w[f'decoder.layers.pre_self_attention_layer_norm_{sub_layer}.scale'][:,_layer]
        state_dict[k] = torch.tensor(v)
        #print(k, v.shape, v.max(), v.min(), v.mean(), v.std())
    model.load_state_dict(state_dict, strict=False)
    return model



if __name__ == '__main__':
    '''
    run command:
    python maxtext2torch.py --model_path  gs://bucket/dcformer/checkpoints --step 14000
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True) # model directory, such as cloud path(gs://bucket/dcformer/checkpoints) or local path(./dcformer/checkpoints)
    parser.add_argument("--step", type=int, required=True) # assign a checkpoint step
    args = parser.parse_args()
    # read metadata and make shape and dtype like checkpoint struct
    
    load_step = args.step 
    read_dir = args.model_path
    print('read_dir', read_dir)
    print('load_step', load_step)
    
    weights = load_model(read_dir, load_step=load_step)

    # initialize a pytorch dcformer model 
    from configuration_dcformer import DCFormerConfig
    from modeling_dcformer import DCFormer
    # DCFormer-Medium
    config = {"vocab_size": 50256,"n_layer": 24, "n_head":16, "head_dim": 128, "dim": 1024, "use_qk_norm": True, "window_type": "LG", "window_size": 256, "rope_base":10000}
    config = DCFormerConfig(**config)
    model = DCFormer(config) 
    print('init dcformer done')

    # convert maxtext model weight to pytorch model weight
    model = update_weight_from_maxtext(model, weights, vocab_size=50256, num_blocks=2, model_dim=1024, num_heads=16, head_dim=128)
    model.save_pretrained("dcformer_medium.pth", safe_serialization=False)
    print('converted')

