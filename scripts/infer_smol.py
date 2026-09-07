"""Run inference with the converted SmolLM2-360M-Instruct weights on the repo's
custom Llama model.

Usage:
    python scripts/infer_smol.py --prompt "What is the capital of France?"
    python scripts/infer_smol.py --chat
    python scripts/infer_smol.py --chat --max-new-tokens 128 --temperature 0.7
"""

import argparse
import os
import sys

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.inference.generate import generate, generate_stream, load_model, load_tokenizer  # noqa: E402
from src.inference.metrics import ServingPerformanceProfiler, format_metrics  # noqa: E402


def _device_and_dtype(device_arg: str):
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda"), torch.bfloat16
        return torch.device("cpu"), torch.float32
    device = torch.device(device_arg)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    return device, dtype


def _chat_input(tokenizer, messages, device: torch.device) -> torch.Tensor:
    out = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    ids = out["input_ids"] if hasattr(out, "get") else out
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)


def run_single(args, model, tokenizer, device):
    input_ids = _chat_input(tokenizer, [{"role": "user", "content": args.prompt}], device)
    out = generate(
        model,
        tokenizer,
        input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    print(tokenizer.decode(out[0], skip_special_tokens=True))


def run_chat(args, model, tokenizer, device):
    messages = []
    print("Interactive chat (type 'quit' to exit).")
    while True:
        user = input("You: ").strip()
        if user.lower() in {"quit", "exit", "q"}:
            break
        messages.append({"role": "user", "content": user})
        input_ids = _chat_input(tokenizer, messages, device)
        out = generate(
            model,
            tokenizer,
            input_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
        reply = tokenizer.decode(out[0, input_ids.size(1):], skip_special_tokens=True)
        messages.append({"role": "assistant", "content": reply})
        print("Assistant:", reply)


def run_benchmark(args, model, tokenizer, device):
    input_ids = _chat_input(tokenizer, [{"role": "user", "content": args.prompt}], device)
    profiler = ServingPerformanceProfiler(device, model=model)
    stream = generate_stream(
        model,
        tokenizer,
        input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    tokens, metrics = profiler.profile_generation_run(stream)
    print(format_metrics(metrics))
    reply = tokenizer.decode(torch.tensor([tokens], dtype=torch.long), skip_special_tokens=True)
    print("\nResponse:\n", reply)


def main():
    parser = argparse.ArgumentParser(description="SmolLM2-360M-Instruct inference on the custom Llama model.")
    parser.add_argument("--model-dir", default=os.path.join("weights", "smol2_360m_instruct"))
    parser.add_argument("--prompt", default=None, help="Single-shot completion prompt.")
    parser.add_argument("--chat", action="store_true", help="Interactive chat REPL.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--benchmark", action="store_true", help="Profile the run and print inference metrics.")
    args = parser.parse_args()

    device, dtype = _device_and_dtype(args.device)
    print(f"==> Loading model + tokenizer from {args.model_dir} (device={device}, dtype={dtype})")
    model = load_model(args.model_dir, device, dtype)
    tokenizer = load_tokenizer(args.model_dir)

    if args.benchmark:
        if not args.prompt:
            parser.error("--benchmark requires --prompt.")
        run_benchmark(args, model, tokenizer, device)
    elif args.chat:
        run_chat(args, model, tokenizer, device)
    elif args.prompt:
        run_single(args, model, tokenizer, device)
    else:
        parser.error("Provide --prompt or --chat.")


if __name__ == "__main__":
    main()
