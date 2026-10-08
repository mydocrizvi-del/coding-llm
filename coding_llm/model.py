from dataclasses import dataclass
import torch
from torch import nn

@dataclass
class ModelConfig:
    vocab_size: int
    max_seq_len: int
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    d_ff: int = 2048
    dropout: float = 0.0
    rope_theta: float = 10000.0

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

def rotate_half(x):
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)

def apply_rope(q, k, cos, sin):
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        if cfg.d_model % cfg.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        inv = 1.0 / (cfg.rope_theta ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("inv_freq", inv, persistent=False)
    def forward(self, x):
        b, t, c = x.shape
        q, k, v = self.qkv(x).split(c, dim=-1)
        q = q.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        pos = torch.arange(t, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(pos, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos, sin = emb.cos()[None, None, :, :], emb.sin()[None, None, :, :]
        q, k = apply_rope(q, k, cos, sin)
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out(y.transpose(1, 2).contiguous().view(b, t, c))

class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.up = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)
    def forward(self, x):
        return self.down(torch.nn.functional.silu(self.gate(x)) * self.up(x))

class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1, self.attn = RMSNorm(cfg.d_model), CausalSelfAttention(cfg)
        self.norm2, self.mlp = RMSNorm(cfg.d_model), MLP(cfg)
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))

class CodingLLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok.weight
        self.apply(self._init_weights)
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    def forward(self, input_ids, labels=None):
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if input_ids.size(1) > self.cfg.max_seq_len:
            raise ValueError(f"sequence length {input_ids.size(1)} exceeds max_seq_len {self.cfg.max_seq_len}")
        x = self.tok(input_ids)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.norm(x))
        loss = None if labels is None else nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), labels.reshape(-1)
        )
        return {"logits": logits, "loss": loss}
    @classmethod
    def from_config(cls, cfg_dict):
        return cls(ModelConfig(**cfg_dict))
    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())
