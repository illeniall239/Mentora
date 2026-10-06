# Reference solution: only used by scripts/verify-exercises.mjs. Shown in the Breakdown only after the Learner's own solution passes.
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, block: int):
        super().__init__()
        self.n_heads = n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.register_buffer("mask", torch.tril(torch.ones(block, block, dtype=torch.bool)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        q, k, v = self.qkv(x).split(D, dim=-1)
        # (B, T, D) -> (B, heads, T, d_head)
        q = q.view(B, T, self.n_heads, -1).transpose(1, 2)
        k = k.view(B, T, self.n_heads, -1).transpose(1, 2)
        v = v.view(B, T, self.n_heads, -1).transpose(1, 2)
        scores = q @ k.transpose(-2, -1) / (q.shape[-1] ** 0.5)
        scores = scores.masked_fill(~self.mask[:T, :T], float("-inf"))  # -inf BEFORE the softmax
        out = F.softmax(scores, dim=-1) @ v
        return self.proj(out.transpose(1, 2).reshape(B, T, D))


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, block: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, block)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))  # pre-LN: normalise the input of each sublayer, add to the residual stream
        return x + self.mlp(self.ln2(x))


class TinyGPT(nn.Module):
    def __init__(self, vocab: int, block: int, d_model: int, n_heads: int, n_layers: int):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.block = block
        self.tok_emb = nn.Embedding(vocab, d_model)
        self.pos_emb = nn.Embedding(block, d_model)
        self.blocks = nn.ModuleList(Block(d_model, n_heads, block) for _ in range(n_layers))
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape
        if T > self.block:
            raise ValueError(f"sequence length {T} exceeds block size {self.block}")
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T))
        for blk in self.blocks:
            x = blk(x)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, ids: torch.Tensor, n: int) -> torch.Tensor:
        for _ in range(n):
            logits, _ = self(ids[:, -self.block :])
            nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            ids = torch.cat([ids, nxt], dim=1)
        return ids
