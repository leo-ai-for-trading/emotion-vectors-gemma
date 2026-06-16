#!/usr/bin/env python3
from __future__ import annotations

import argparse

import torch

from emotion_vectors_gemma.modeling import load_model, load_tokenizer
from emotion_vectors_gemma.steering import steer_residual_stream
from emotion_vectors_gemma.vectors import load_vector_set


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text while steering with an emotion vector.")
    parser.add_argument("--model-id")
    parser.add_argument("--vectors", required=True)
    parser.add_argument("--emotion", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--strength", type=float, default=0.05)
    parser.add_argument("--layers", help="Comma-separated decoder-layer indices. Default: middle three.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    vector_set = load_vector_set(args.vectors)
    model_id = args.model_id or vector_set.model_id
    tokenizer = load_tokenizer(model_id)
    model, device, _dtype = load_model(model_id, device=args.device, dtype=args.dtype)
    direction = vector_set.vector_for(args.emotion, normalize=True)
    layers = None
    if args.layers:
        layers = [int(value.strip()) for value in args.layers.split(",") if value.strip()]

    encoded = tokenizer(args.prompt, return_tensors="pt").to(device)
    with torch.inference_mode(), steer_residual_stream(
        model=model,
        direction=direction,
        strength=args.strength,
        layers=layers,
    ):
        generated = model.generate(
            **encoded,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    print(tokenizer.decode(generated[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()

