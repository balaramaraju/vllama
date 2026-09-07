"""Download SmolLM2-360M(-Instruct) and convert its weights into the repo's
custom ``Llama`` layout.

The Hugging Face checkpoint uses the standard ``LlamaForCausalLM`` naming scheme
(``model.embed_tokens``, ``model.layers.*``, ``lm_head``); the repo model uses
``embedding`` / ``blocks.*`` / ``norm`` names. This script maps one onto the
other, saves a single ``model.safetensors`` plus a small ``config.json`` (our
LlamaConfig fields) and the tokenizer, then optionally verifies our ported
model's logits against the HF reference model.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

# Make the `src` package importable when this file is run directly.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from safetensors.torch import save_file  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from src.inference.smol_config import DEFAULT_MODEL_ID, SMOL2_360M, make_smol2_config  # noqa: E402
from src.model.llama import Llama  # noqa: E402


def _pick_device_and_dtype() -> tuple[torch.device, torch.dtype]:
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.bfloat16
    return torch.device("cpu"), torch.float32


def hf_to_repo_state_dict(hf_sd: dict, n_blocks: int) -> dict:
    """Map ``LlamaForCausalLM`` state-dict keys to the repo ``Llama`` keys."""
    sd: dict = {}
    sd["embedding.weight"] = hf_sd["model.embed_tokens.weight"]
    for i in range(n_blocks):
        p = f"model.layers.{i}."
        b = f"blocks.{i}."
        sd[b + "attention.q_proj.weight"] = hf_sd[p + "self_attn.q_proj.weight"]
        sd[b + "attention.k_proj.weight"] = hf_sd[p + "self_attn.k_proj.weight"]
        sd[b + "attention.v_proj.weight"] = hf_sd[p + "self_attn.v_proj.weight"]
        sd[b + "attention.o_proj.weight"] = hf_sd[p + "self_attn.o_proj.weight"]
        sd[b + "attention_norm.weights"] = hf_sd[p + "input_layernorm.weight"]
        sd[b + "ffn_norm.weights"] = hf_sd[p + "post_attention_layernorm.weight"]
        sd[b + "feed_forward.w1.weight"] = hf_sd[p + "mlp.gate_proj.weight"]
        sd[b + "feed_forward.w2.weight"] = hf_sd[p + "mlp.down_proj.weight"]
        sd[b + "feed_forward.w3.weight"] = hf_sd[p + "mlp.up_proj.weight"]
    sd["norm.weights"] = hf_sd["model.norm.weight"]
    # Embeddings are tied, so `lm_head.weight` is the same tensor as
    # `model.embed_tokens.weight`. Fall back just in case the key is absent.
    sd["output.weight"] = hf_sd.get("lm_head.weight", hf_sd["model.embed_tokens.weight"])
    return sd


def verify_logits(repo_model, ref_model, tokenizer, device, dtype) -> bool:
    """Compare our ported model's logits to the HF reference on a short prompt."""
    prompt = "The capital of France is"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        ref_logits = ref_model(input_ids).logits
        repo_logits, _ = repo_model(input_ids)

    max_diff = (repo_logits - ref_logits).abs().max().item()
    atol = 2e-2 if dtype == torch.bfloat16 else 1e-4
    rtol = 5e-2 if dtype == torch.bfloat16 else 1e-4
    ok = torch.allclose(repo_logits, ref_logits, atol=atol, rtol=rtol)
    print(f"    max |delta logits| = {max_diff:.6f}  (atol={atol}, rtol={rtol})")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download + convert SmolLM2-360M weights for the custom Llama model."
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--out-dir", default=os.path.join("weights", "smol2_360m_instruct"))
    parser.add_argument("--dtype", choices=["auto", "bf16", "fp32"], default="auto")
    parser.add_argument("--no-verify", action="store_true", help="Skip logit comparison vs HF reference.")
    args = parser.parse_args()

    device, auto_dtype = _pick_device_and_dtype()
    dtype = {"auto": auto_dtype, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]

    print(f"==> Loading HF reference: {args.model_id} (dtype={dtype}, device={device})")
    ref_model = (
        AutoModelForCausalLM.from_pretrained(
            args.model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        .to(device)
        .eval()
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    print("==> Building repo Llama model...")
    config = make_smol2_config()
    repo_model = Llama(config=config, use_activation_checkpoint=False).to(device)
    if dtype != torch.float32:
        repo_model = repo_model.to(dtype)
    repo_model.eval()

    hf_sd = ref_model.state_dict()
    repo_sd = hf_to_repo_state_dict(hf_sd, config.n_blocks)

    missing = set(repo_model.state_dict().keys()) - set(repo_sd.keys())
    unexpected = set(repo_sd.keys()) - set(repo_model.state_dict().keys())
    if missing or unexpected:
        raise RuntimeError(
            f"Key mismatch — missing={sorted(missing)} unexpected={sorted(unexpected)}"
        )

    repo_model.load_state_dict(repo_sd)
    n_params = sum(p.numel() for p in repo_model.parameters())
    print(f"==> Loaded {len(repo_sd)} tensors ({n_params:,} params) into repo model.")

    os.makedirs(args.out_dir, exist_ok=True)
    tokenizer.save_pretrained(os.path.join(args.out_dir, "tokenizer"))
    # Tied embeddings: `output.weight` shares memory with `embedding.weight`.
    # safetensors refuses duplicate/shared tensors, so drop the tied copy.
    sd = repo_model.state_dict()
    sd.pop("output.weight", None)
    save_file(sd, os.path.join(args.out_dir, "model.safetensors"))
    with open(os.path.join(args.out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(SMOL2_360M, f, indent=2)

    print(f"==> Saved model.safetensors + config.json + tokenizer -> {args.out_dir}")

    if not args.no_verify:
        print("==> Verifying logits vs HF reference...")
        if verify_logits(repo_model, ref_model, tokenizer, device, dtype):
            print("   PASS")
        else:
            print("   FAIL")
            sys.exit(1)


if __name__ == "__main__":
    main()

