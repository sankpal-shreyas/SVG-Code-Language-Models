"""Decoder-only transformer. Adapted from karpathy/nanoGPT (commit dbb6b3b).

Modifications from upstream nanoGPT:
- attn_scale_mode flag toggles 1/sqrt(d_head) vs 1/d_head for muP.
- mup_readout flag swaps lm_head for mup.MuReadout.
- Drops the GPT-2 weight loader, the optimizer factory, and Flash CUDA fallbacks.
- YAML-driven config dataclass instead of CLI overrides.
"""
import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 4096
    block_size: int = 1024
    n_layer: int = 4
    n_head: int = 4
    d_model: int = 128
    d_ff: int = 512
    dropout: float = 0.0
    bias: bool = False
    tie_weights: bool = False
    attn_scale_mode: str = "sp"
    mup_readout: bool = False
    init_std: float = 0.02


class LayerNorm(nn.Module):
    def __init__(self, d: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.bias = nn.Parameter(torch.zeros(d)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.d_head = cfg.d_model // cfg.n_head
        self.c_attn = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.dropout = cfg.dropout
        if cfg.attn_scale_mode == "mup":
            self.scale = 1.0 / self.d_head
        else:
            self.scale = 1.0 / math.sqrt(self.d_head)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
            scale=self.scale,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(cfg.d_model, cfg.d_ff, bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.d_ff, cfg.d_model, bias=cfg.bias)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.c_proj(F.gelu(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = LayerNorm(cfg.d_model, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = LayerNorm(cfg.d_model, bias=cfg.bias)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = LayerNorm(cfg.d_model, bias=cfg.bias)

        if cfg.mup_readout:
            from mup import MuReadout
            self.lm_head = MuReadout(cfg.d_model, cfg.vocab_size, bias=False, readout_zero_init=True)
        else:
            self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        if cfg.tie_weights:
            if cfg.mup_readout:
                raise ValueError("tie_weights is not supported with mup_readout")
            self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            try:
                from mup import MuReadout
                if isinstance(m, MuReadout):
                    return
            except ImportError:
                pass
            nn.init.normal_(m.weight, mean=0.0, std=self.cfg.init_std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=self.cfg.init_std)

    def num_params(self, include_embeddings: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if not include_embeddings:
            n -= self.tok_emb.weight.numel()
            n -= self.pos_emb.weight.numel()
        return n

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"sequence length {T} > block_size {self.cfg.block_size}"
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if targets is None:
            return logits, None
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=-100,
        )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None, eos_id=None):
        for _ in range(max_new_tokens):
            cond = idx[:, -self.cfg.block_size:]
            logits, _ = self(cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, -1:]] = -float("inf")
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cum = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                mask = cum > top_p
                mask[..., 1:] = mask[..., :-1].clone()
                mask[..., 0] = False
                unsorted_mask = torch.zeros_like(mask)
                unsorted_mask.scatter_(-1, sorted_idx, mask)
                logits = logits.masked_fill(unsorted_mask, -float("inf"))
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, nxt], dim=1)
            if eos_id is not None and (nxt == eos_id).all():
                break
        return idx
