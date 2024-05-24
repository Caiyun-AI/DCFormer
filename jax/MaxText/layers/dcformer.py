import os
from typing import Optional

import jax
from flax import linen as nn
from jax.sharding import Mesh
import jax.numpy as jnp

from layers import dc_attentions
from layers import embeddings
from layers import linears
from layers import normalizations
from layers import models
from layers import initializers
import tensorflow as tf

if os.environ["HARDWARE"] == "gpu":
    Quant = None
else:
    from layers import quantizations
    Quant = quantizations.AqtQuantization

import common_types


Array = common_types.Array
Config = common_types.Config
DType = common_types.DType
Mesh = common_types.Mesh
ScanIn = common_types.ScanIn

Embed = embeddings.Embed
Attention = dc_attentions.Attention
RMSNorm = normalizations.RMSNorm
NormalInitializer = initializers.nd_dense_init_normal

#-----------------------------------------
# The Decoder Layer specific for Dcformer++
#-----------------------------------------


class DcformerDecoderLayer(nn.Module):
  """Transformer decoder layer that attends to the encoder."""
  config: models.Config
  mesh: Mesh
  quant: Optional[Quant] = None

  @nn.compact
  def __call__(self,
               inputs,
               decoder_segment_ids,
               decoder_positions,
               deterministic,
               model_mode,
               num_layers_per_block=None,
               ):
    num_layers_per_block = 1 if num_layers_per_block is None else int(num_layers_per_block)
    window_size = self.config.window_size
    if window_size is None:
        window_size = [None]
    elif isinstance(window_size, list):
        for size in window_size:
            assert isinstance(size, int) or size is None, print(f'window_size value error: {size}')
    else:
        raise ValueError(f'Window size: ‘{window_size}’ type is error.....')

    for i in range(num_layers_per_block):
        layer_output = self.sub_block(inputs, decoder_segment_ids, decoder_positions, deterministic, model_mode, window_size[i], i)
        inputs = layer_output[0] if self.config.scan_layers else layer_output

    return layer_output
        
  def sub_block(self,
               inputs,
               decoder_segment_ids,
               decoder_positions,
               deterministic,
               model_mode,
               window_size,
               block_index,
               ):
    cfg = self.config
    mesh = self.mesh

    inputs = nn.with_logical_constraint(
        inputs, ('activation_batch', 'activation_length', 'activation_embed')) # fsdp, 1, mp


    lnx_rms = models.RMSNorm(
        dtype=cfg.dtype,
        name=f'pre_self_attention_layer_norm_{block_index}',
        kernel_axes=('embed',),
        epsilon=cfg.normalization_layer_epsilon,
        )
    lnx = lnx_rms(inputs)

    lnx = nn.with_logical_constraint(
        lnx, ('activation_batch', 'activation_length', 'activation_embed'))

    assert cfg.attention == 'dot_product', print(f'Now dcformer model only support ’dot_product‘ method to compute attention')
    # Self-attention block
    attention_layer = Attention(
      config = cfg,
      num_query_heads=cfg.num_query_heads,
      num_kv_heads=cfg.num_kv_heads,
      head_dim=cfg.head_dim,
      max_target_length=cfg.max_target_length,
      max_prefill_predict_length=cfg.max_prefill_predict_length,
      attention_kernel=cfg.attention,
      mesh=mesh,
      dtype=cfg.dtype,
      dropout_rate=cfg.dropout_rate,
      name=f'self_attention_{block_index}',
      float32_qk_product = False,  # computes logits in float32 for stability.
      float32_logits = True,
      quant=self.quant,
      window_size=window_size,
      kernel_init=NormalInitializer(0.006),
      )

    attention_lnx = attention_layer(
            lnx,
            lnx,
            decoder_positions,
            decoder_segment_ids=decoder_segment_ids,
            deterministic=deterministic,
            model_mode=model_mode)

    attention_lnx = nn.with_logical_constraint(
        attention_lnx,
        ('activation_batch', 'activation_length', 'activation_embed'))
    intermediate_inputs = inputs + attention_lnx

    # Fully Connected
    hidden_states = models.RMSNorm(
        dtype=cfg.dtype, name=f'sub.{block_index}.post_self_attention_layer_norm', kernel_axes=('embed',),
        epsilon=cfg.normalization_layer_epsilon,
        )(intermediate_inputs)
    hidden_states = nn.with_logical_constraint(hidden_states, ('activation_batch', 'activation_length', 'activation_embed'))

    # MLP block.
    mlp_lnx = linears.MlpBlock(
        intermediate_dim=cfg.mlp_dim,
        activations=cfg.mlp_activations,
        intermediate_dropout_rate=cfg.dropout_rate,
        dtype=cfg.dtype,
        name=f'sub.{block_index}.mlp',
        config=cfg,
        quant=self.quant,
        kernel_init=NormalInitializer(0.006),
    )(hidden_states, deterministic=deterministic)
    mlp_lnx = nn.with_logical_constraint(
        mlp_lnx, ('activation_batch', 'activation_length', 'activation_embed')
    )

    layer_output = mlp_lnx + intermediate_inputs

    layer_output = nn.Dropout(
        rate=cfg.dropout_rate, broadcast_dims=(-2,))(
            layer_output, deterministic=deterministic)

    layer_output = nn.with_logical_constraint(
        layer_output,
        ('activation_batch', 'activation_length', 'activation_embed'),
    )

    if cfg.record_internal_nn_metrics:
      self.sow('intermediates', 'activation_mean', jnp.mean(layer_output))
      self.sow('intermediates', 'activation_stdev', jnp.std(layer_output))
      self.sow(
          'intermediates',
          'activation_fraction_zero',
          jnp.sum(layer_output == 0) / jnp.size(layer_output),
      )

    if cfg.scan_layers:
      return layer_output, None
    else:
      return layer_output
