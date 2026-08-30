import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from torch.utils.checkpoint import checkpoint

class LlamaGroupAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.n_embd // config.n_heads
        self.num_queries_per_kv = self.n_heads // self.n_kv_heads
        
        self.q_proj = nn.Linear(config.n_embd, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.n_embd, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.n_embd, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, config.n_embd, bias=False)
        
        self.rope = SimpleRoPE(dim=self.head_dim, max_seq_len=config.max_seq_len)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, s, c = x.shape
        
        q = self.q_proj(x).view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, s, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, s, self.n_kv_heads, self.head_dim).transpose(1, 2)
        
        q = self.rope(q, seq_len=s)
        k = self.rope(k, seq_len=s)
        
        if self.num_queries_per_kv > 1:
            k = k.repeat_interleave(self.num_queries_per_kv, dim=1)
            v = v.repeat_interleave(self.num_queries_per_kv, dim=1)
            
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores + mask.unsqueeze(0).unsqueeze(1)
            
        scores = F.softmax(scores, dim=-1).to(x.dtype)
        
        output = torch.matmul(scores, v)
        output = output.transpose(1, 2).contiguous().view(b, s, c)
        
        return self.o_proj(output)

class LlamaBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = LlamaGroupAttention(config=config)
        self.attention_norm = RMSNorm(dim=config.n_embd)
        self.ffn_norm = RMSNorm(dim=config.n_embd)
        self.feed_forward = SwiGLUMLP(config=config)

    def forward(self, tokens: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        tokens = tokens + self.attention(self.attention_norm(tokens), mask=mask)
        tokens = tokens + self.feed_forward(self.ffn_norm(tokens))
        return tokens

class SimpleRoPE(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 2048, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.inv_freq: torch.Tensor
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached: torch.Tensor
        self.sin_cached: torch.Tensor
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        cos = self.cos_cached[:seq_len, :].unsqueeze(0).unsqueeze(1)
        sin = self.sin_cached[:seq_len, :].unsqueeze(0).unsqueeze(1)
        return (x * cos) + (self._rotate_half(x) * sin)

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weights = nn.Parameter(torch.ones(dim))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        org_dtype = tokens.dtype
        tokens = tokens.to(torch.float32)
        variance = tokens.pow(2).mean(-1, keepdim=True)
        norm_tokens = tokens * torch.rsqrt(variance + self.eps)
        return norm_tokens.to(org_dtype) * self.weights

class SwiGLUMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = self.get_hidden_dim(config.n_embd)
        self.w1 = nn.Linear(config.n_embd, self.hidden_dim, bias=False)
        self.w2 = nn.Linear(self.hidden_dim, config.n_embd, bias=False)
        self.w3 = nn.Linear(config.n_embd, self.hidden_dim, bias=False)

    @staticmethod
    def get_hidden_dim(n_embd: int) -> int:
        hidden_dim = int(2 * (n_embd * 4) / 3)
        hidden_dim = ((hidden_dim + 255) // 256) * 256
        return hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Llama(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = nn.ModuleList(LlamaBlock(config) for _ in range(config.n_blocks))
        self.norm = RMSNorm(dim=config.n_embd)
        self.output = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.output.weight = self.embedding.weight

    def forward(self, tokens: torch.Tensor, targets: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_len = tokens.shape
        x = self.embedding(tokens)
        
        mask = torch.full((seq_len, seq_len), float("-inf"), device=tokens.device)
        mask = torch.triu(mask, diagonal=1)

        # ---------------------------------------------------------------------
        # PERFORMANCE OPTIMIZATION: ACTIVATION CHECKPOINT
        # Converts keyword parameters to clean positional arguments to satisfy
        # the use_reentrant=False non-reentrant FSDP2 graph constraint.
        # ---------------------------------------------------------------------
        def create_checkpoint_forward(block_layer):
            def custom_forward(tensor_state, attention_mask):
                # Unpacks safely inside the localized block frame
                return block_layer(tensor_state, mask=attention_mask)
            return custom_forward

        for block in self.blocks:
            # Drop intermediate activations (SwiGLU, QKV dots) during forward pass;
            # force on-the-fly recomputation during the backward graph step.
            x = checkpoint(
                create_checkpoint_forward(block),
                x,
                mask,
                use_reentrant=False  # Mandatory for FSDP2 tracking integration
            )
        # ---------------------------------------------------------------------

        x = self.norm(x)
        logits = self.output(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                targets.view(-1), 
                ignore_index=-1
            )
            
        return logits, loss

if __name__ == "__main__":
    class ModelConfig:
        vocab_size = 32000
        n_embd = 512
        n_blocks = 2
        n_heads = 8
        n_kv_heads = 2
        max_seq_len = 2048

    config = ModelConfig()
    model = Llama(config)
    
    mock_tokens = torch.randint(0, config.vocab_size, (4, 16))
    mock_targets = torch.randint(0, config.vocab_size, (4, 16))
    
    logits, loss = model(mock_tokens, targets=mock_targets)
    print("Execution Success!")
    print("Logits Tensor Shape:", logits.shape)
    print("Calculated Training Loss Value:", loss.item())
