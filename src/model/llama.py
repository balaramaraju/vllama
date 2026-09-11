import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from torch.utils.checkpoint import checkpoint
from src.kvcache import KVCacheMemory

class LlamaGroupAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.n_kv_heads: int = config.n_kv_heads
        self.head_dim = config.n_embd // config.n_heads
        self.num_queries_per_kv = self.n_heads // self.n_kv_heads
        
        self.q_proj = nn.Linear(config.n_embd, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.n_embd, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.n_embd, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, config.n_embd, bias=False)
        
        self.rope = SimpleRoPE(dim=self.head_dim, max_seq_len=config.max_seq_len, theta=config.rope_theta)

    def forward(
    self, 
    x: torch.Tensor, 
    position_ids: Optional[torch.Tensor] = None,  # Added for Phase 1 RoPE fix
    mask: Optional[torch.Tensor] = None, 
    kv_memory: Optional[KVCacheMemory] = None,  # The physical flat tensor pool
    slot_mapping: Optional[torch.Tensor] = None,# Tells us EXACTLY where to write tokens
    block_tables: Optional[torch.Tensor] = None,# Tells us which blocks belong to which sequence
    context_lens: Optional[torch.Tensor] = None,# Total steps (past history + current token) per batch item
    layer_idx: int = 0                          # Identifies this block's layer in the cache pool
    ) -> torch.Tensor:
        b, s, c = x.shape

        # Phase 1: position_ids is optional so training/smoke-test call sites that
        # don't supply it keep working unchanged. When omitted, fall back to the
        # contiguous 0..s-1 positions within the current sequence.
        if position_ids is None:
            position_ids = torch.arange(s, dtype=torch.long, device=x.device).unsqueeze(0).repeat(b, 1)
        
        q = self.q_proj(x).view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, s, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, s, self.n_kv_heads, self.head_dim).transpose(1, 2)
        
        q = self.rope(q, position_ids)
        k = self.rope(k, position_ids)

        if kv_memory is not None and slot_mapping is not None:
            # k/v shapes coming in: (b, n_kv_heads, s, head_dim)
            # We transpose to (b, s, n_kv_heads, head_dim) and flatten batch * sequence length into total tokens
            k_flat = k.transpose(1,2).reshape(-1, self.n_kv_heads , self.head_dim)
            v_flat = v.transpose(1,2).reshape(-1, self.n_kv_heads,self.head_dim)

            # Write to physical cache pages
            kv_memory.write_kv(k_flat,v_flat, slot_mapping=slot_mapping, layer_idx=layer_idx)

        # Since we have no customized Paged-CUDA attention kernel yet, standard eager torch.matmul
        # requires data to be contiguous in memory. We use the block_tables to look up physical pages
        # and assemble the past K/V context history back into chronological arrays.

        if kv_memory is not None and block_tables is not None and context_lens is not None:
            max_context_len = torch.max(context_lens).item()
            k_storage, v_storage = kv_memory.get_key_value(layer_idx=layer_idx)

            # Pre-allocate temporary clean buffers
            k_assembled = torch.zeros(
                (int(b), int(self.n_kv_heads), int(max_context_len), int(self.head_dim)), 
                dtype=x.dtype, 
                device=x.device
            )
            v_assembled = torch.zeros(
                (int(b), int(self.n_kv_heads), int(max_context_len), int(self.head_dim)), 
                dtype=x.dtype, 
                device=x.device
            )

            # Reconstruct timelines token-by-token using virtual layout rules
            #
            # TODO(#4): This O(context) Python double loop + full dense matmul per
            # decode step defeats the purpose of the block table — it rebuilds
            # k_assembled/v_assembled from pages every step with fresh allocations
            # and is likely slower than a naive full-forward. The real paged-attention
            # win requires vectorizing the gather (a single index_select/gather per
            # step using block_tables) or a dedicated paged-attention kernel that
            # reads directly out of kv_memory without assembling dense KV.
            for i in range(b):
                cur_seq_len = context_lens[i].item()
                for pos in range(int(cur_seq_len)):
                    logical_block_idx = pos // kv_memory.block_size
                    block_offset = pos % kv_memory.block_size
                    physical_block_idx = block_tables[i, logical_block_idx].item()
                    
                    # Read out from flat block array structure
                    k_assembled[i, :, pos, :] = k_storage[int(physical_block_idx), block_offset, :, :]
                    v_assembled[i, :, pos, :] = v_storage[int(physical_block_idx), block_offset, :, :]
                    
            k, v = k_assembled, v_assembled

        
        if self.num_queries_per_kv > 1:
            k = k.repeat_interleave(self.num_queries_per_kv, dim=1)
            v = v.repeat_interleave(self.num_queries_per_kv, dim=1)

        k = k.contiguous()
        v = v.contiguous()
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores + mask
            
        scores = F.softmax(scores, dim=-1).to(x.dtype)
        
        output = torch.matmul(scores, v)
        output = output.transpose(1, 2).contiguous().view(b, s, c)
        
        return self.o_proj(output)

class LlamaBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = LlamaGroupAttention(config=config)
        self.attention_norm = RMSNorm(dim=config.n_embd, eps=config.rms_norm_eps)
        self.ffn_norm = RMSNorm(dim=config.n_embd, eps=config.rms_norm_eps)
        self.feed_forward = SwiGLUMLP(config=config)

    def forward(
        self, 
        tokens: torch.Tensor, 
        position_ids: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None, 
        kv_memory: Optional[KVCacheMemory] = None, 
        slot_mapping: Optional[torch.Tensor] = None, 
        block_tables: Optional[torch.Tensor] = None, 
        context_lens: Optional[torch.Tensor] = None, 
        layer_idx: int = 0
    ) -> torch.Tensor:
        # This level also needs its own fallback: the activation-checkpointing
        # branch (see Llama.forward) invokes blocks via `custom_forward` without
        # passing position_ids, so we must not require it here. Use shape[0]/[1]
        # so both raw token-ids (2D) and hidden states (3D) are handled.
        if position_ids is None:
            b, s = tokens.shape[0], tokens.shape[1]
            position_ids = torch.arange(s, dtype=torch.long, device=tokens.device).unsqueeze(0).repeat(b, 1)

        tokens = tokens + self.attention(
            self.attention_norm(tokens), 
            position_ids=position_ids, 
            mask=mask, 
            kv_memory=kv_memory, 
            slot_mapping=slot_mapping, 
            block_tables=block_tables, 
            context_lens=context_lens, 
            layer_idx=layer_idx
        )
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

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, num_heads, seq_len, head_dim)
            position_ids: Tensor of shape (batch_size, seq_len) containing the absolute positions
        """
        # Gather position values from precomputed cache maps
        # Resulting shape: (batch_size, seq_len, head_dim)
        cos = self.cos_cached[position_ids]
        sin = self.sin_cached[position_ids]
        
        # Unsqueeze to align with num_heads dimension -> (batch_size, 1, seq_len, head_dim)
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        
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
    def __init__(self, config, use_activation_checkpoint: bool = True):
        super().__init__()
        self.config = config
        self.use_activation_checkpoint = use_activation_checkpoint
        self.embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = nn.ModuleList(LlamaBlock(config) for _ in range(config.n_blocks))
        self.norm = RMSNorm(dim=config.n_embd, eps=config.rms_norm_eps)
        self.output = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.output.weight = self.embedding.weight

    def forward(
        self, 
        tokens: torch.Tensor, 
        position_ids: Optional[torch.Tensor] = None,  # Added for Phase 1 RoPE
        targets: Optional[torch.Tensor] = None,
        kv_memory: Optional[KVCacheMemory] = None,   # Added for Phase 3 Paged Cache
        slot_mapping: Optional[torch.Tensor] = None, # Added for Phase 3 Paged Cache
        block_tables: Optional[torch.Tensor] = None, # Added for Phase 3 Paged Cache
        context_lens: Optional[torch.Tensor] = None  # Added for Phase 3 Paged Cache
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        batch_size, seq_len = tokens.shape

        # Phase 1: training and the __main__ smoke test call forward without
        # position_ids. When omitted, derive the contiguous 0..seq_len-1
        # positions internally so those call sites keep working unchanged.
        if position_ids is None:
            position_ids = torch.arange(seq_len, dtype=torch.long, device=tokens.device).unsqueeze(0).repeat(batch_size, 1)

        x = self.embedding(tokens)

        # --------------------------------------------------------------------
        # STEP 3.1: DYNAMIC RECTANGULAR MASK GENERATION
        # --------------------------------------------------------------------
        if context_lens is not None:
            # Mask needs to match the max historical tokens present across the current batch items.
            # Each batch item may have stopped early (e.g. hit EOS) and so has fewer valid history
            # tokens than the padded max. Initialize everything to -inf so those trailing padding
            # columns are fully masked (attending to zero-filled paddings would otherwise dilute the
            # softmax), then overlay the per-item causal pattern on only each item's valid columns.
            max_context_len = int(torch.max(context_lens).item())
            mask = torch.full(
                (int(batch_size), 1, int(seq_len), max_context_len),
                float("-inf"),
                device=tokens.device,
                dtype=x.dtype,
            )

            for i in range(int(batch_size)):
                cur_seq_len = int(context_lens[i].item())
                # Mask out positions ahead of the token's respective timeline
                causal_mask = torch.full((int(seq_len), cur_seq_len), float("-inf"), device=tokens.device).triu(diagonal=1 + (cur_seq_len - int(seq_len)))
                mask[i, 0, :, :cur_seq_len] = causal_mask
        else:
            # Fallback to standard square causal matrix if no memory layout is active
            mask = torch.full((int(seq_len), int(seq_len)), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=1)

        # --------------------------------------------------------------------
        # STEP 3.2: PASSTHROUGH TO BLOCKS (Bypassing activation checkpointing if caching)
        # --------------------------------------------------------------------
        # Activation Checkpointing drops states to minimize backward pass footprints.
        # This will corrupt cache pointers during generation, so force eager path if caching.
        if self.use_activation_checkpoint and kv_memory is None:
            def create_checkpoint_forward(block_layer):
                def custom_forward(tensor_state, attention_mask):
                    return block_layer(tensor_state, mask=attention_mask)
                return custom_forward

            for block in self.blocks:
                x = checkpoint(
                    create_checkpoint_forward(block),
                    x,
                    mask,
                    use_reentrant=False,
                )
        else:
            # Pass all memory parameters down loop layer by layer
            for layer_idx, block in enumerate(self.blocks):
                x = block(
                    x, 
                    position_ids=position_ids, 
                    mask=mask, 
                    kv_memory=kv_memory, 
                    slot_mapping=slot_mapping, 
                    block_tables=block_tables, 
                    context_lens=context_lens, 
                    layer_idx=layer_idx
                )

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
        rope_theta = 10000.0
        rms_norm_eps = 1e-6

    config = ModelConfig()
    model = Llama(config)
    
    mock_tokens = torch.randint(0, config.vocab_size, (4, 16))
    mock_targets = torch.randint(0, config.vocab_size, (4, 16))
    
    logits, loss = model(mock_tokens, targets=mock_targets)
    print("Execution Success!")
    print("Logits Tensor Shape:", logits.shape)
    print("Calculated Training Loss Value:", loss.item())
