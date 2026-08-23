from dataclasses import dataclass

@dataclass
class LlamaConfig:
    vocab_size: int = 128256
    n_embd: int = 1536
    n_blocks: int = 16
    n_heads: int = 24
    n_kv_heads: int = 8
    max_seq_len: int = 2048


@dataclass
class LocalLlamaConfig:
    vocab_size: int = 128256
    n_embd: int = 256
    n_blocks: int = 2
    n_heads: int = 2
    n_kv_heads: int = 1
    max_seq_len: int = 256