# Assemble + Use Radial Attention

# imports
import torch, json, pathlib, math, random, copy
from pathlib import Path
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from peft import LoraConfig, get_peft_model
from transformers import DecisionTransformerConfig, GPT2Config, GPT2LMHeadModel
from transformer.art import AutonomousRendezvousTransformer # get actual model
import os, sys
import shutil
from torch.optim import AdamW
from accelerate import Accelerator
from radial_attn.attn_mask import RadialAttention, MaskMap # from actual repo

BLOCK=128

class RadialSelfAttention1D(nn.Module):
  """
    Create RadialSelfAttention1D layer.
  """
  def __init__(self, embed_dim, num_heads, radial_window, bias=True):
    super().__init__()
    self.embed_dim = embed_dim
    self.num_heads = num_heads
    self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=bias)
    self.out = nn.Linear(embed_dim, embed_dim, bias=bias)
    self.mask_map = MaskMap(video_token_num=0, num_frame=1)

  def copy_from_dense(self, dense_block):
    # weight+bias radial blocks
    with torch.no_grad():
      self.qkv.weight.copy_(dense_block.c_attn.weight.T)
      self.qkv.bias.copy_(dense_block.c_attn.bias)
      self.out.weight.copy_(dense_block.c_proj.weight.T)
      self.out.bias.copy_(dense_block.c_proj.bias)

  def forward(self, x, **_):
    B, T, D = x.shape
    qkv = self.qkv(x).view(B, T, 3, self.num_heads, D // self.num_heads)
    q, k, v = [
    t.permute(1, 0, 2, 3)          # (T, B, H, hdim)
     .reshape(T * B, self.num_heads, -1)
    for t in qkv.permute(2, 1, 0, 3, 4)
]

    # call actual radial attention - i think hunyuan vs. wan wont matter?
    y = RadialAttention(q, k, v,
                        mask_map=self.mask_map,
                        sparsity_type="radial",
                        block_size=BLOCK,
                        decay_factor=1.0,
                        model_type="hunyuan")
    y = y.view(T, B, self.num_heads, -1).permute(1, 0, 2, 3).reshape(B, T, D)
    y = self.out(y)
    return (y, None, None) # formatting output as was throwing errors
# in place swapping dense attn for radial
def radial_swap(model, w0=16, keep_dense=2):
  """
    Swap dense attn for radial in time, leaving keep_dense layers dense - defaults to boundary layers dense.
  """

  # exclude dense layers
  # for each block size radial block
  for i, block in enumerate(model.encoder.h):
    if i < keep_dense:
      continue
    dense_block = block.attn
    rad_block = RadialSelfAttention1D(
      embed_dim=dense_block.embed_dim,
      num_heads=dense_block.num_heads,
      radial_window=w0,
    )

    rad_block.copy_from_dense(dense_block) # helper

    # swap block
    block.attn = rad_block