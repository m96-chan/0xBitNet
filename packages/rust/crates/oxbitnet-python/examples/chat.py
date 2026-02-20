"""Interactive chat CLI for oxbitnet.

Usage:
    python chat.py
    python chat.py --url /path/to/model.gguf
    python chat.py --temperature 0.5 --max-tokens 1024
"""

import argparse
import sys
import time

from oxbitnet import BitNet

DEFAULT_URL = (
    "https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-gguf/resolve/main/ggml-model-i2_s.gguf"
)


def main():
    parser = argparse.ArgumentParser(description="oxbitnet — interactive BitNet chat")
    parser.add_argument("--url", default=DEFAULT_URL, help="Model GGUF URL or path")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--repeat-penalty", type=float, default=1.1)
    args = parser.parse_args()

    print("oxbitnet — Python CLI", file=sys.stderr)
    print("=====================\n", file=sys.stderr)
    print(f"Loading model from:\n  {args.url}\n", file=sys.stderr)

    model = BitNet.load_sync(args.url)

    print("Model loaded! Type your message (Ctrl+D to exit).\n", file=sys.stderr)

    history = [("system", "You are a helpful assistant.")]

    while True:
        try:
            line = input("You: ")
        except (EOFError, KeyboardInterrupt):
            break

        text = line.strip()
        if not text:
            continue

        history.append(("user", text))

        print("\nAssistant: ", end="", flush=True)

        t0 = time.perf_counter()

        token_count = model.chat(
            history,
            on_token=lambda t: print(t, end="", flush=True),
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            repeat_penalty=args.repeat_penalty,
        )

        elapsed = time.perf_counter() - t0
        tok_sec = token_count / elapsed if elapsed > 0 else 0

        print(f"\n\n  [{token_count} tokens in {elapsed:.1f}s — {tok_sec:.1f} tok/s]\n")

        # Keep system + last 2 turns
        if len(history) > 5:
            history = history[:1] + history[-2:]

    print("\nBye!", file=sys.stderr)
    model.dispose()


if __name__ == "__main__":
    main()
