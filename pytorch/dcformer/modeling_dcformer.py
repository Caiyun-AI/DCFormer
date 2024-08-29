from dataclasses import dataclass
from typing import Optional,Tuple,List
from collections import namedtuple

import math
import time
import json
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

try:
    from .configuration_dcformer import DCFormerConfig
except:
    from configuration_dcformer import DCFormerConfig

from transformers.modeling_utils import PreTrainedModel


class KVKWCache(nn.Module):
    def __init__(self, max_batch_size, max_seq_length, n_heads, head_dim, window_size=2048, dtype=torch.float16, use_kw_cache=True):
        super().__init__()
        self.head_dim = head_dim
        self.kw_dim = 2 * n_heads 
        self.n_heads = n_heads
        self.window_size = window_size
        self.use_kw_cache = use_kw_cache 
        if window_size is None:
            self.seq_length = max_seq_length
        else:
            self.seq_length = min(window_size, max_seq_length)
        cache_shape = (max_batch_size, n_heads, self.seq_length, head_dim)
        kw_cache_shape = (max_batch_size, self.seq_length, 2, n_heads, n_heads)
        self.register_buffer('k_cache', torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer('v_cache', torch.zeros(cache_shape, dtype=dtype))
        if self.use_kw_cache:
            self.register_buffer('kw_cache', torch.zeros(kw_cache_shape, dtype=dtype))

    def update(self, input_pos, k_val, v_val, kw_val=None): # kw_val B,N,S,2,N      B2NSD
        # input_pos: [S], k_val: [B, H, S, D]
        assert input_pos.shape[-1] == k_val.shape[2]
        B,N,S,D = v_val.shape
        k_out = self.k_cache
        v_out = self.v_cache
        if self.use_kw_cache:
            kw_out = self.kw_cache
        else:
            kw_out = None

        if self.window_size is None:
            k_out[:, :, input_pos] = k_val
            v_out[:, :, input_pos] = v_val
            if self.use_kw_cache and kw_val is not None:
                kw_out[:,input_pos] = kw_val
        elif S == 1: 
            input_pos = input_pos % self.seq_length
            v_out[:, :, input_pos] = v_val
            k_out[:, :, input_pos] = k_val
            if self.use_kw_cache and kw_val is not None:
                kw_out[:,input_pos] = kw_val
        else: # prefill
            start = max(0, input_pos[-1]-self.seq_length+1)
            input_pos = input_pos[start:] % self.seq_length
            v_out[:, :, input_pos] = v_val[:,:,start:]
            k_out[:, :, input_pos] = k_val[:,:,start:]
            if self.use_kw_cache and kw_val is not None:
                kw_out[:, input_pos] = kw_val[:,start:]
        return k_out, v_out, kw_out 

class DCFormer(PreTrainedModel):
    config_class=DCFormerConfig
    '''
    DCFormer's implementation is adapted from https://github.com/pytorch-labs/gpt-fast/blob/main/model.py#L89 
    '''

    def __init__(self, config: DCFormerConfig) -> None:
        super().__init__(config)
        self.config = config

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList(DCFormerBlock(config, lidx) for lidx in range(config.n_layer))
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.use_gradient_checkpointing = config.use_gradient_checkpointing 
        self.is_training = config.is_training

        self.freqs_cis: Optional[Tensor] = None
        self.mask_cache: Optional[Tensor] = None
        self.window_size = config.window_size
        self.max_batch_size = -1
        self.max_seq_length = -1

    def setup_caches(self, max_batch_size, max_seq_length, set_kv_cache=True):
        if self.max_seq_length >= max_seq_length and self.max_batch_size >= max_batch_size:
            return
        head_dim = self.config.dim // self.config.n_head
        max_seq_length = find_multiple(max_seq_length, 8)
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        if not self.is_training:
            for b in self.layers:
                if set_kv_cache:
                    use_kw_cache = False if b.attention.query_wise else True
                    b.attention.kv_cache = KVKWCache(max_batch_size, max_seq_length, self.config.n_local_heads, head_dim, window_size=b.attention.window_size, use_kw_cache=use_kw_cache)
                b.attention.dyn_w_proj.merge_weights()
                if not b.attention.use_sw:
                    dtype = b.attention.wo.weight.dtype
                    device = b.attention.wo.weight.device
                    b.attention.dyn_w_proj.sw = b.attention.dyn_w_proj.sw.to(device=device, dtype=dtype)
                    b.attention.dyn_w_proj.pre_proj.w = b.attention.dyn_w_proj.pre_proj.w.to(device=device, dtype=dtype) 
                    b.attention.dyn_w_proj.post_proj.w = b.attention.dyn_w_proj.post_proj.w.to(device=device, dtype=dtype) 
                
        self.freqs_cis = precompute_freqs_cis(self.config.block_size, self.config.dim // self.config.n_head, self.config.rope_base).to(self.tok_embeddings.weight.device)
        if self.is_training:
            self.causal_mask = torch.tril(torch.ones(self.config.block_size, self.config.block_size, dtype=torch.bool, device=self.tok_embeddings.weight.device))
        elif self.window_size is None:
            self.causal_mask = torch.tril(torch.ones(max_seq_length, max_seq_length, dtype=torch.bool, device=self.tok_embeddings.weight.device))
        else:
            self.causal_mask = torch.stack([make_window_mask(max_seq_length, self.config.window_size), torch.tril(torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool))]) # LG
    
    def generate(self, input_ids, num_tokens_to_generate=10, compiled_decode_one_token=None):
        batch_size, seq_length = input_ids.shape
        input_pos = torch.arange(seq_length, device=self.device)
        generated_ids = torch.zeros(batch_size, seq_length + num_tokens_to_generate + 1, dtype=torch.int, device=self.device)
        generated_ids[:, :seq_length] = input_ids.to(self.device).to(torch.int)
        logits = self.forward(input_ids, input_pos=input_pos,return_tensor=True)
        _next_token = torch.argmax(logits[:, -1], dim=-1)[:, None]
        next_token = torch.zeros(self.max_batch_size, 1, device=self.device, dtype=torch.int)
        next_token[:batch_size] = _next_token
        generated_ids[:, seq_length] = next_token[:batch_size, 0]
        input_pos = torch.tensor([seq_length], device=self.device)
        for _ in range(1, num_tokens_to_generate):
            if compiled_decode_one_token is not None:
                next_token = compiled_decode_one_token(self, next_token.clone(), input_pos)
            else:
                next_token = self.decode_one_token(next_token.clone(), input_pos)
            generated_ids[:, input_pos+1] = next_token.int()[:batch_size]
            input_pos += 1
        return generated_ids
    
    def decode_one_token(self, cur_token, input_pos):
        logits = self.forward(
            cur_token,
            input_pos=input_pos,
            return_tensor=True
        )
        new_token = torch.argmax(logits[:, -1], dim=-1)[:,None]
        return new_token

    def forward(self, idx: Tensor, input_pos: Optional[Tensor] = None, return_tensor=False) -> Tensor:
        assert self.freqs_cis is not None, "Caches must be initialized first"
        if input_pos is None:
            input_pos = torch.arange(idx.shape[-1], device=idx.device, dtype=torch.int)
        if self.window_size is None or self.is_training:
            mask = self.causal_mask[None, None, input_pos]
        else:
            mask = self.causal_mask[None, None,:,input_pos]
        freqs_cis = self.freqs_cis[input_pos][:idx.shape[-1]]
        x = self.tok_embeddings(idx)
        for i, layer in enumerate(self.layers):
            if self.is_training or self.window_size is None :
                layer_mask = mask
            elif self.window_size is not None: 
                layer_mask = mask[:,:,1] if layer.attention.window_size is None else mask[:,:,0]
            if self.use_gradient_checkpointing:
                x = checkpoint(layer, x, input_pos, freqs_cis, layer_mask)
            else:
                x = layer(x, input_pos, freqs_cis, layer_mask)
        x = self.norm(x)
        logits = self.output(x)
        if return_tensor:
            return logits
        else:
            CausalLMOutput = namedtuple("CausalLMOutput", ["logits"])
            return CausalLMOutput(logits=logits)

class DCFormerBlock(nn.Module):
    def __init__(self, config: DCFormerConfig, lidx) -> None:
        super().__init__()
        self.lidx = lidx
        self.attention = DCMHAttention(config, lidx)
        self.feed_forward = FeedForward(config)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)
        self.attention_norm = RMSNorm(config.dim, config.norm_eps)

    def forward(self, x: Tensor, input_pos: Tensor, freqs_cis: Tensor, mask: Tensor) -> Tensor:
        h = x + self.attention(self.attention_norm(x), freqs_cis, mask, input_pos, fast_infer=True)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

class DynamicWeightProjection(nn.Module):

    def __init__(self, num_heads=32, num_groups=1, residual=True, query_input_dim=4096, dynamic_squeeze_ratio=16, dynamic_w_hidden_dim=128,dtype=torch.float16,use_sw=False):
        super().__init__()
        self.num_heads = num_heads 
        self.num_groups = num_groups 
        self.query_input_dim = query_input_dim 
        self.dynamic_squeeze_ratio = dynamic_squeeze_ratio
        self.dynamic_w_hidden_dim = dynamic_w_hidden_dim 
        self.dw_hidden_activation = nn.GELU()
        self.num_heads_per_group = self.num_heads // self.num_groups
        self.dw_activation = nn.Tanh()
        self.dw1_norm = RMSnormNoscale(dim=-1)
        self.use_sw = use_sw
        self.pre_proj = CrossHeadProjection('pre', num_heads=self.num_heads, use_sw=use_sw)
        self.post_proj = CrossHeadProjection('post', num_heads=self.num_heads, use_sw=use_sw)

        dynamic_hidden_dim = self.num_heads_per_group // self.dynamic_squeeze_ratio 
        self.dynamic_hidden_dim = dynamic_hidden_dim 
        self.dw1 = nn.parameter.Parameter(torch.zeros(self.query_input_dim, self.num_groups, 4, self.dynamic_w_hidden_dim, dtype=dtype)) #(4096, 1, 4, 128)
        G, K, M = self.num_groups, self.dynamic_w_hidden_dim, self.num_heads_per_group
        I = dynamic_hidden_dim * 2 
        self.qkw = nn.parameter.Parameter(torch.zeros([G, 4, K, I, M], dtype=dtype)) # (1, 4, 128, 4, 32)
        self.dd = nn.parameter.Parameter(torch.zeros(self.query_input_dim, self.num_groups, self.num_heads_per_group * 4, dtype=dtype)) #  (4096, 1, 128)

        self.merge_weights()

    def merge_weights(self):
        self.dw_m = nn.parameter.Parameter(torch.cat([self.dw1.reshape(self.query_input_dim, -1), self.dd.squeeze(1)], dim=-1)).to(self.dw1.device) # E,(4*K + K)  K=2*N*I
        self.qkw_m = nn.parameter.Parameter(self.qkw.permute(0,1,2,3,4).reshape(4,self.dynamic_w_hidden_dim,-1)).to(self.dw1.device) #(4,K,I*M)
        if self.use_sw:
            self.sw = nn.parameter.Parameter(torch.stack([self.pre_proj.w, self.post_proj.w]).squeeze(1) + torch.eye(self.num_heads) ).to(self.dw1.device) # (2,N,N) sw + identity matrix
        else:
            self.sw = (torch.eye(self.num_heads).expand(2,self.num_heads,self.num_heads)).to(self.dw1.device) # identity matrix (2,N,N)
        
    def forward(self,query_vec,KW:Optional[torch.Tensor]=None, gen_cache:Optional[bool]=True):  
        dw_hidden = torch.einsum('BTD,DGCK->BTGCK', query_vec, self.dw1)  # C=4 [pre,post]*[query,key]
        dw_hidden = self.dw_hidden_activation(dw_hidden) #BTGCK
        w1, w2 = torch.split(torch.einsum('BTGCK,GCKIM->BTGCIM', dw_hidden, self.qkw), self.qkw.shape[-2]//2, dim=-2) #BTGC(2I)M -> [BTGCIM] * 2
        w1 = self.dw1_norm(w1) # BTGCIM
        pre_qw1, pre_kw1, post_qw1, post_kw1 = unbind(w1, 4, dim=3) # BTG4IM->[BTGIM]*4
        pre_qw2, pre_kw2, post_qw2, post_kw2 = unbind(w2, 4, dim=3) 
        dd = torch.einsum('BTD,DGM->BTGM', query_vec, self.dd) # BTG(4M)
        dd = self.dw_activation(dd)
        pre_qdd, pre_kdd, post_qdd, post_kdd = torch.split(dd, dd.shape[-1] // 4, dim=-1) # BTG(4N)->[BTGN]*4
        pre_dw_args = (pre_qw1, pre_qw2, pre_kw1, pre_kw2, pre_qdd, pre_kdd)
        post_dw_args = (post_qw1, post_qw2, post_kw1, post_kw2, post_qdd, post_kdd)
        if gen_cache: # generate KW cache
            pre_kw = torch.einsum('BSGIM, BSGIN->BSMN', pre_kw1, pre_kw2) + torch.diag_embed(pre_kdd.squeeze(2))  # merge kw and kdd
            post_kw = torch.einsum('BSGIM, BSGIN->BSMN', post_kw1, post_kw2) + torch.diag_embed(post_kdd.squeeze(2))
            KW = torch.stack((pre_kw, post_kw), dim=-3) # BSMN,BSMN->BS2MN
        return pre_dw_args, post_dw_args, KW


class RMSnormNoscale(nn.Module):
    
    def __init__(self, epsilon=1e-6, dim=-1):
        super().__init__()
        self.dim = dim 
        self.epsilon = epsilon

    def forward(self, inputs):
        var = inputs.pow(2).mean(dim=self.dim, keepdim=True)
        normed_inputs = inputs * torch.rsqrt(var + self.epsilon)
        return normed_inputs 


class RMSnorm(nn.Module):

    def __init__(self, hid_dim=128, epsilon=1e-6, dim=-1):
        super().__init__()
        self.dim = dim
        self.hid_dim = hid_dim
        self.epsilon = epsilon
        self.scale = nn.parameter.Parameter(data=torch.ones(self.hid_dim))

    def forward(self, inputs):
        var = inputs.pow(2).mean(dim=self.dim, keepdim=True)
        normed_inputs = inputs * torch.rsqrt(var + self.epsilon)
        normed_inputs = normed_inputs * self.scale
        return normed_inputs


class CrossHeadProjection(nn.Module):

    def __init__(self, mode, num_heads=16, num_groups=1, dtype=torch.float16, use_sw=False):
        super().__init__()
        self.mode = mode
        self.use_sw = use_sw
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.num_heads_per_group = self.num_heads // self.num_groups
        if self.use_sw:
            self.w = nn.parameter.Parameter(data=torch.zeros(self.num_groups, self.num_heads_per_group, self.num_heads_per_group, dtype=dtype))
        else:
            self.register_buffer('w', torch.eye(self.num_heads_per_group, dtype=dtype).expand(self.num_groups, self.num_heads_per_group, self.num_heads_per_group))

    def forward(self, inputs, 
            dws:Optional[Tuple[Tensor,Tensor, Tensor,Tensor, Tensor,Tensor]]=None,
            query_vec=None, key_vec=None, 
            proj_w:Optional[Tensor]=None,
            fast_infer=True): 
        if proj_w is not None: 
            ret = torch.einsum('BNTS,BSNM->BMTS', inputs, proj_w)
        else:
            assert dws is not None
            qw1, qw2, kw1, kw2, qdd, kdd = dws
            inputs = inputs.unsqueeze(1) #BNTS->BGNTS
            # apply sw 
            ret = torch.einsum('BGMTS,GMN->BGNTS', inputs, self.w) + inputs if self.use_sw else inputs
            if fast_infer:
                inputs_label = 'BGMTS'
                hidden_sym = 'I'; hidden_label = inputs_label.replace('M', 'I') # BGITS
                # apply qw and kw
                for sym, (w1, w2) in zip(['T', 'S'], [(qw1, qw2), (kw1, kw2)]): 
                    dw_label = f'B{sym}G{hidden_sym}M'  # w1: BTGIM, dw_label:BTGIM
                    dynamic_hidden_dim = w1.shape[dw_label.index(hidden_sym)]
                    eqn1 = f'{inputs_label},{dw_label}->{hidden_label}' # 'BGMTS,BTGMI->BGITS'
                    eqn2 = f'{hidden_label},{dw_label}->{inputs_label}' # 'BGITS,BTGMI->BGMTS'
                    for i in range(dynamic_hidden_dim):
                        hidden = torch.einsum(eqn1.replace(hidden_sym, ''), inputs, w1[..., i, :]) # BGMTS,BTG(I)M->BGTS
                        out = torch.einsum(eqn2.replace(hidden_sym, ''), hidden, w2[..., i, :]) #  'BG(I)TS,BTG(I)M->BGMTS'
                        ret = ret + out
                # apply qdd and kdd
                for sym, dd in zip(['T', 'S'], [qdd, kdd]):
                    dd_label = f'B{sym}GM'
                    dout = torch.einsum(f'{inputs_label},{dd_label}->{inputs_label}', inputs, dd) # BGMTS,B(T/S)GM->BGMTS
                    ret = ret + dout
            else:
                # apply qw and kw (BTGIN)
                x_inter = torch.einsum('BGNTS, BTGIN->BGTSI', inputs, qw1)
                qw_out = torch.einsum('BGTSI, BTGIN->BGNTS', x_inter, qw2)
                ret = ret + qw_out
                x_inter = torch.einsum('BGNTS, BSGIN->BGTSI', inputs, kw1)
                kw_out = torch.einsum('BGTSI, BSGIN->BGNTS', x_inter, kw2)
                ret = ret + kw_out

                # apply qdd(BTGN) and kdd(BSGN)
                ret = ret + torch.einsum('BGNTS, BTGN->BGNTS', inputs, qdd)
                ret = ret + torch.einsum('BGNTS, BSGN->BGNTS', inputs, kdd)
            ret = ret.squeeze(1) # BGNTS->BNTS    
        return ret  


class DCMHAttention(nn.Module):
    def __init__(self, config: DCFormerConfig, lidx, use_sw=False):
        super().__init__()
        assert config.dim % config.n_head == 0
        total_head_dim = (config.n_head + 2 * config.n_local_heads) * config.head_dim
        # key, query, value projections for all heads, but in a batch
        self.lidx = lidx
        self.wqkv = nn.Linear(config.dim, total_head_dim, bias=False)
        self.wo = nn.Linear(config.dim, config.dim, bias=False)
        self.kv_cache = None

        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.n_local_heads = config.n_local_heads
        self.is_training = config.is_training
        self.dim = config.dim
        self.use_dcmha = config.use_dcmha 
        self.scale_factor = 1 / math.sqrt(self.head_dim)
        self.q_chunk_size = config.q_chunk_size 
        self.use_sw = use_sw 
        self.dyn_w_proj = DynamicWeightProjection(num_heads=self.n_head, query_input_dim=config.dim, dynamic_squeeze_ratio=self.n_head//2, dynamic_w_hidden_dim=self.n_head*4, use_sw=use_sw)
        self.use_qk_norm = config.use_qk_norm 
        if self.use_qk_norm:
            self.q_norm = RMSnorm(hid_dim=self.head_dim)
            self.k_norm = RMSnorm(hid_dim=self.head_dim)

        self.window_types = {
            "LG":[256, None],
            "LGLL":[256, None, 256, 256],
            "LGL6":[256, None, 256, 256, 256, 256, 256, 256],
        }

        self.query_wise = config.query_wise
        if config.window_type is None: # LG 
            self.window_size = None if self.lidx % 2 == 1 else config.window_size 
        else:
            window_l = self.window_types[config.window_type]
            self.window_size = window_l[self.lidx % len(window_l)]

        if not self.is_training:
            self._register_load_state_dict_pre_hook(self.load_hook)

    def load_hook(self, state_dict, prefix, *args):
        if prefix + "wq.weight" in state_dict:
            wq = state_dict.pop(prefix + "wq.weight")
            wk = state_dict.pop(prefix + "wk.weight")
            wv = state_dict.pop(prefix + "wv.weight")
            state_dict[prefix + "wqkv.weight"] = torch.cat([wq, wk, wv])
   
    def _generate_fast(self, x, input_pos, q, k, v, k_mask):
        B,T,D = x.shape
        N,I = self.n_head, self.dyn_w_proj.dynamic_hidden_dim # 32, 2
        dw_hidden, dd = (x @ self.dyn_w_proj.dw_m).split([2*2*N*(2*I), 2*2*N*1], -1) # BTD, D(4K+4N) -> BT(4K+4N) -> BT(4K), BT(4N)
        dw_hidden = dw_hidden.view((B,T,4,-1,1)) # BT(4K) -> BT4K1
        dw = (self.dyn_w_proj.dw_hidden_activation(dw_hidden) * self.dyn_w_proj.qkw_m).sum(-2) # gelu, BT4K1, 4K(IM)->BT4K(IM)->BT4(IM)
        w1, w2 = dw.view((B,T,2,2,-1,N)).split(I,-2) # BT4(IM)->BT{pre/post}{q/k}IM->[BT22IM] * 2
        w1 = self.dyn_w_proj.dw1_norm(w1) # BT22IN
        qkdd = self.dyn_w_proj.dw_activation(dd.view((B,T,2,2,N))) # BT2{2}N1->BT2{2}N tanh
        qkw = torch.einsum('BTKJIN,BTKJIM->BTKJNM', w1, w2) + torch.diag_embed(qkdd) # j=k=2, BT2{2}NM q/k, pre/post
        if self.query_wise: # TODO: do not generate kw and kdd
            qw, _ = qkw.unbind(3) # BS2NM
            kw_new = None
            qw = qw + self.dyn_w_proj.sw 
        else:
            qw, kw_new = qkw.unbind(3) # BS{pre/post}{q/k}NM -> BS{pre/post}NM * 2
            kw_new = kw_new + self.dyn_w_proj.sw  # BS2NM + 2NM-> BS2NM 
        if self.kv_cache is not None:
            k, v, kw_out = self.kv_cache.update(input_pos, k, v, kw_val=kw_new) #BNT2M
        logits = q @ k.transpose(-2, -1) * self.scale_factor 
        if self.query_wise:
            w = qw  # B12NM
        else:
            w = qw + kw_out # B12NM,BS2NM -> BS2NM 
        wl, w = w.permute(0,2,3,4,1).unbind(1)  # BS2NM->B2NMS->[BNMS]*2 
        logits = (logits * wl).sum(1).unsqueeze(2) # BN1S, BNMS -> BNMS-> BMS-> BM1S 
        min_value = torch.finfo(torch.float16).min
        logits = torch.where(k_mask, logits, min_value)
        probs = logits.softmax(-1)
        probs = (probs * w).sum(1).unsqueeze(2)
        y = probs @ v
        return y

    def forward(self, x: Tensor, freqs_cis: Tensor, mask: Tensor, input_pos: Optional[Tensor] = None, fast_infer=True) -> Tensor:
        bsz, seqlen, _ = x.shape

        kv_size = self.n_local_heads * self.head_dim
        q, k, v = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)

        q = q.view(bsz, seqlen, self.n_head, self.head_dim) # BSND
        k = k.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        if self.use_qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)

        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v)) # BNSD

        if self.is_training:
            N, D, I = self.n_head, self.head_dim, self.dyn_w_proj.dynamic_hidden_dim; # 6.7B
            B,T,E = x.shape
            if self.use_dcmha:
                project_logits = True 
                project_probs = True
                if project_probs:
                    dw_hidden, dd = (x @ self.dyn_w_proj.dw_m).split([2*2*N*(2*I), 2*2*N*1], -1)
                    dw_hidden = self.dyn_w_proj.dw_hidden_activation(dw_hidden) 
                    dw_hidden = dw_hidden.view(dw_hidden.shape[:2]+(4,-1)) #B T (4 K) -> B T 4 K  # reshape
                    dw = torch.einsum('B T C K, C K D -> B T C D', dw_hidden, self.dyn_w_proj.qkw_m) # BT4K,4K(MI)->BT4(MI)
                    shape = (B,T,2*2,-1,N)# if project_logits else (B,T,2,N,-1)  # BT(pre/post)(q/k)IN
                    w1, w2 = dw.view(shape).split(I,-2)
                    w1 = self.dyn_w_proj.dw1_norm(w1) # BT22IN
                    if self.use_sw:
                        pre_sw, post_sw = self.dyn_w_proj.sw.unbind(0)
                    else:
                        pre_sw, post_sw = None, None
                    pre_qw1, pre_kw1, post_qw1, post_kw1 = w1.unbind(2)  # BT(2{*2})IN->[BTIN]*4
                    pre_qw2, pre_kw2, post_qw2, post_kw2 = w2.unbind(2)
                    qkdd = F.tanh(dd).squeeze(-1).view(shape[:-2] + (N,)) # BT(2{*2})N1->BT(2{*2})N
                    pre_qdd, pre_kdd, post_qdd, post_kdd = qkdd.unbind(2)  # BT(2{*2})N->[BTN]*4

                y = torch.zeros(B, N, T, D).to(q.device, dtype=torch.float16)
                for i in range(T // self.q_chunk_size):
                    start, stop = i * self.q_chunk_size, (i + 1) * self.q_chunk_size
                    kv_start = max(0, stop - self.q_chunk_size -self.window_size)
                    _q = q[:, :, start : stop, :]
                    _k, _v = k[:, :, kv_start : stop, :], v[:, :, kv_start : stop, :]
                    _atten_mask = mask[:, :, start : stop, kv_start : stop]
                    _pre_proj_dw_args = slice_dw(pre_sw, pre_qw1, pre_qw2, pre_kw1, pre_kw2, pre_qdd, pre_kdd, start, stop, kv_start) \
                        if project_logits else None
                    _post_proj_dw_args = slice_dw(post_sw, post_qw1, post_qw2, post_kw1, post_kw2, post_qdd, post_kdd, start,stop,kv_start) \
                        if project_probs else None
                    _o = _atten_context(_q, _k, _v, _atten_mask, _pre_proj_dw_args, _post_proj_dw_args)
                    y[:,:,start:stop] = _o
            else:
                y = torch.zeros(B, N, T, D).to(q.device, dtype=torch.float16)
                for i in range(T // self.q_chunk_size):
                    start, stop = i * self.q_chunk_size, (i + 1) * self.q_chunk_size
                    kv_start = max(0, stop - self.q_chunk_size -self.window_size)
                    _q = q[:, :, start : stop, :]
                    _k, _v = k[:, :, kv_start : stop, :], v[:, :, kv_start : stop, :]
                    _atten_mask = mask[:, :, start : stop, kv_start : stop]
                    _pre_proj_dw_args, _post_proj_dw_args = None, None
                    _o = _atten_context(_q, _k, _v, _atten_mask, _pre_proj_dw_args, _post_proj_dw_args)
                    y[:,:,start:stop] = _o
        else: # inference
            if seqlen == 1: # one-token generation
                k_mask = mask if self.window_size is None else mask[:,:,:,:self.kv_cache.seq_length]
                if fast_infer:
                    y = self._generate_fast(x, input_pos, q, k, v, k_mask)
                else: 
                    assert not self.query_wise
                    # generate dw from hidden_state
                    pre_proj_dw_args, post_proj_dw_args, kw_new = self.dyn_w_proj(x, gen_cache=True)
                    
                    # update kvkw cache
                    kw_new = kw_new + self.dyn_w_proj.sw # absorb residual or sw into kw cache
                    if self.kv_cache is not None:
                        k, v, kw_out = self.kv_cache.update(input_pos, k, v, kw_val=kw_new) # BNSD, BNSD, BS2NN

                    logits = q @ k.transpose(-2, -1) * self.scale_factor
                    # merge pre_w and apply it
                    pre_qw1, pre_qw2, pre_kw1, pre_kw2, pre_qdd, pre_kdd = pre_proj_dw_args
                    pre_qw = torch.einsum('BTGIN, BTGIM->BTNM',pre_qw1, pre_qw2)  + torch.diag_embed(pre_qdd.squeeze(2))
                    pre_w = pre_qw + kw_out[:,:,0] # B1NM, BSNM -> BSNM
                    logits = self.dyn_w_proj.pre_proj(logits, proj_w=pre_w.squeeze(1))
  
                    logits = torch.where(k_mask, logits, torch.finfo(torch.float16).min)
                    probs = logits.softmax(-1)

                    # merge post_w and apply it
                    post_qw1, post_qw2, post_kw1, post_kw2, post_qdd, post_kdd = post_proj_dw_args
                    post_qw = torch.einsum('BTGIN, BTGIM->BTNM', post_qw1, post_qw2) + torch.diag_embed(post_qdd.squeeze(2))
                    post_w = post_qw + kw_out[:,:,1]
                    probs = self.dyn_w_proj.post_proj(probs, proj_w=post_w.squeeze(1)) 

                    y = probs @ v                  
            else: # prefill
                k_mask = mask[:,:,:,:k.shape[-2]] 
                pre_proj_dw_args, post_proj_dw_args,kw_new = self.dyn_w_proj(x, gen_cache=True)
                kw_new = kw_new + self.dyn_w_proj.sw # absorb residual or sw into kw cache
                if self.kv_cache is not None:
                    self.kv_cache.update(input_pos, k, v, kw_val=kw_new) # BNSD, BNSD, BS2NN
                logits = q @ k.transpose(-2, -1) * self.scale_factor 
                logits = self.dyn_w_proj.pre_proj(logits, dws=pre_proj_dw_args, query_vec=x, key_vec=x, fast_infer=True)  # XD BN1S
                logits = torch.where(k_mask, logits, torch.finfo(torch.float16).min)
                probs = logits.softmax(-1)
                probs = self.dyn_w_proj.post_proj(probs, dws=post_proj_dw_args, query_vec=x, key_vec=x, fast_infer=True) # BN1S
                y = probs @ v

        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)
        y = self.wo(y)
        return y


class FeedForward(nn.Module):
    def __init__(self, config: DCFormerConfig) -> None:
        super().__init__()
        self.w1 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w3 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def _atten_context(query, key, value, atten_mask, pre_proj_dw_args, post_proj_dw_args):
    logits = query @ key.transpose(-2, -1)
    if pre_proj_dw_args is not None: logits = _cross_head_proj(logits, *pre_proj_dw_args)
    logits = torch.where(atten_mask, logits, torch.finfo(torch.float16).min)
    probs = logits.softmax(-1)
    if post_proj_dw_args is not None: probs = _cross_head_proj(probs, *post_proj_dw_args)
    o = probs @ value  # BNTS,BNSD->BNTD
    return o

def _cross_head_proj(inputs, sw, qw1, qw2, kw1, kw2, qdd, kdd, loop_over_dynamic_hd=False):
    out = inputs + torch.einsum('BNTS,NM->BMTS', inputs, sw) if sw is not None else inputs
    for i in range(2): # qw1.shape[-2]):
        qhidden = (inputs * qw1[..., i, :].transpose(-2, -1).unsqueeze(-1)).sum(1)  # BNTS,(BTN->BNT->BNT1)->BNTS->BTS
        qout = qhidden.unsqueeze(1) * qw2[..., i, :].transpose(-2, -1).unsqueeze(-1) # (BTS->B1TS),(BTN->BNT->BNT1)->BNTS
        out = out + qout
        khidden = (inputs * kw1[..., i, :].transpose(-2, -1).unsqueeze(-2)).sum(1)  # BNTS,(BSN->BNS->BN1S)->BNTS->BTS
        kout = khidden.unsqueeze(1) * kw2[..., i, :].transpose(-2, -1).unsqueeze(-2) # (BTS->B1TS),(BSN->BNS->BNS1)->BNTS
        out = out + kout
    qdout = inputs * qdd.transpose(-2, -1).unsqueeze(-1); out = out + qdout  # BNTS,(BTN->BNT->BNT1)->BNTS
    kdout = inputs * kdd.transpose(-2, -1).unsqueeze(-2); out = out + kdout  # BNTS,(BSN->BNS->BN1S)->BNTS
    return out

def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)

def make_window_mask(t, window_size):
    col_idx = torch.tile(torch.arange(t).unsqueeze(0), [t, 1])
    row_idx = torch.tile(torch.arange(t).unsqueeze(1), [1, t])
    bias_mask = (col_idx + window_size >= row_idx).tril().view(t, t)
    return bias_mask 

def slice_dw(sw, qw1, qw2, kw1, kw2, qdd, kdd, start, stop, kv_start):
    return (sw,
            qw1[:, start : stop] if qw1 is not None else None,
            qw2[:, start : stop] if qw2 is not None else None,
            kw1[:, kv_start : stop] if kw1 is not None else None,
            kw2[:, kv_start : stop] if kw2 is not None else None,
            qdd[:, start : stop] if qdd is not None else None,
            kdd[:, kv_start : stop] if kdd is not None else None)

def precompute_freqs_cis(
    seq_len: int, n_elem: int, base: int = 10000
) -> Tensor:
    freqs = 1.0 / (base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem))
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    return cache.to(dtype=torch.float16)

def unbind(ary, n, dim=0):
    return [torch.squeeze(a, dim=dim) for a in torch.split(ary, ary.shape[dim] // n, dim=dim)]

def apply_rotary_emb(x: Tensor, freqs_cis: Tensor, mode='half') -> Tensor:
    if mode == 'half':
        xshaped = x.float().reshape(*x.shape[:-1], 2,-1).transpose(-1,-2) 
    elif mode == 'alternative':
        xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    freqs_cis = freqs_cis.view(-1, xshaped.size(1), 1, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
        ],
        -1,
    )
    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)
