"""
gated_ssm_agent.py
Simplified version for Birdie music generation without flex_attention.
"""

import torch
import einops
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from birdie_rl.birdie_reward_model import rotary

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("medium")

softmax = nn.Softmax(dim=-1)
sigmoid = nn.Sigmoid()

default_max_seq_len = 2048


class RMSNorm(nn.Module):
    def __init__(self, dims, scale_init_fn=torch.ones, eps=1e-8):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.scale = nn.Parameter(scale_init_fn(dims))

    def forward(self, x):
        norm = x.norm(2, dim=-1, keepdim=True) * (x.shape[-1] ** -0.5)
        return self.scale * x / (norm + self.eps)


class RMS_split(nn.Module):
    def __init__(self, input_dim, output_dim=None, dropout_rate=0.0, eps=1e-8):
        super(RMS_split, self).__init__()
        self.norm = RMSNorm(dims=input_dim, eps=eps)
        self.scale = nn.Parameter(torch.zeros(input_dim) + 0.1)
        self.output_dim = output_dim
        self.dropout = nn.Dropout(dropout_rate)
        if self.output_dim is not None:
            self.out = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, x):
        x = self.norm(x)
        x = x * self.scale
        if self.output_dim is None:
            return x
        return self.out(x)


class SwiGLU(nn.Module):
    def __init__(self, dims, hidden_dims):
        super(SwiGLU, self).__init__()
        dims = int(dims)
        hidden_dims = int(hidden_dims)
        self.input_dims = dims
        self.hidden_dims = hidden_dims
        self.wi = nn.Linear(dims, hidden_dims * 2, bias=False)
        self.wo = nn.Linear(hidden_dims, dims, bias=False)
        self.norm = RMS_split(dims)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.wi(x)
        (main, gate) = x.chunk(2, dim=-1)
        x = main * torch.sigmoid(gate)
        return self.wo(x) + residual


class MHA(nn.Module):
    def __init__(self, dims, head_dims=64, freqs_cis=None):
        super(MHA, self).__init__()
        self.num_heads = dims // head_dims
        self.gqa_num_heads = 2  # Just kept as it was
        q_dims = head_dims * self.num_heads
        v_dims = head_dims * self.gqa_num_heads
        self.freqs_cis = freqs_cis

        self.q_proj = nn.Linear(dims, q_dims, bias=False)
        self.k_proj = nn.Linear(dims, v_dims, bias=False)
        self.v_proj = nn.Linear(dims, v_dims, bias=False)
        self.o_proj = nn.Linear(q_dims, dims, bias=False)
        self.norm = RMS_split(dims)

    def forward(self, x, block_mask=None):
        residual = x
        x = self.norm(x)

        q = einops.rearrange(self.q_proj(x), 'b s (h d) -> b s h d', h=self.num_heads)
        k = einops.rearrange(self.k_proj(x), 'b s (h d) -> b s h d', h=self.gqa_num_heads)
        if self.freqs_cis is not None:
            q, k = rotary.apply_rotary_emb(q, k, self.freqs_cis)
        q = einops.rearrange(q, 'b s h d -> b h s d')
        k = einops.rearrange(k, 'b s h d -> b h s d')
        v = einops.rearrange(self.v_proj(x), 'b s (h d) -> b h s d', h=self.gqa_num_heads)

        # Use simple dot-product attention (basic)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (q.shape[-1] ** 0.5)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_out = torch.matmul(attn_weights, v)

        attn_out = einops.rearrange(attn_out, 'b h s d -> b s (h d)')
        mha_out = self.o_proj(attn_out)
        return residual + mha_out


class MLPModel(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dims=[64, 64],
        dropout_rate=0.0,
        num_heads=1,
        max_seq_len=default_max_seq_len,
        device=None,
    ):
        super(MLPModel, self).__init__()

        self.layers = nn.ModuleList()
        if len(hidden_dims) >= 2:
            self.head_dims = hidden_dims[-2] // 4
        else:
            self.head_dims = 64

        self.freqs_cis = rotary.precompute_freqs_cis(
            self.head_dims,
            max_seq_len,
            use_scaled=(max_seq_len != default_max_seq_len),
        ).to(device) if device else None

        self.sequence_length = max_seq_len

        self.layers.append(RMS_split(input_dim, hidden_dims[0], dropout_rate=dropout_rate))
        for i in range(len(hidden_dims)):
            self.layers.append(
                MHA(
                    dims=hidden_dims[i],
                    head_dims=self.head_dims,
                    freqs_cis=self.freqs_cis,
                )
            )
            self.layers.append(SwiGLU(hidden_dims[i], hidden_dims[i] * (8/3)))

        self.layers.append(RMS_split(hidden_dims[-1]))
        self.layers.append(nn.Linear(hidden_dims[-1], output_dim, bias=False))

    def forward(self, x, **kwargs):
        for layer in self.layers:
            x = layer(x)
        return torch.tanh(x)
