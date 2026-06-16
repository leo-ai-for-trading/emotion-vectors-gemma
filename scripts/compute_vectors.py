#!/usr/bin/env python3
from __future__ import annotations

import argparse

from emotion_vectors_gemma.activations import mean_hidden_activations
from emotion_vectors_gemma.datasets import read_jsonl, story_texts, validate_story_records
from emotion_vectors_gemma.modeling import load_model, load_tokenizer, resolve_hidden_state_index
from emotion_vectors_gemma.vectors import build_vector_set, save_vector_set


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute paper-style emotion vectors.")
    parser.add_argument("--model-id", default="google/gemma-3-270m")
    parser.add_argument("--stories", required=True)
    parser.add_argument("--neutral")
    parser.add_argument("--out", required=True)
    parser.add_argument("--layer", default="auto")
    parser.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--start-token", type=int, default=50)
    parser.add_argument("--neutral-variance-threshold", type=float, default=0.5)
    parser.add_argument("--skip-neutral-pca", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    story_records = read_jsonl(args.stories)
    validate_story_records(story_records)

    tokenizer = load_tokenizer(args.model_id)
    model, device, dtype = load_model(args.model_id, device=args.device, dtype=args.dtype)
    layer_index = resolve_hidden_state_index(model, args.layer)

    story_activations = mean_hidden_activations(
        model=model,
        tokenizer=tokenizer,
        texts=story_texts(story_records),
        layer_index=layer_index,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        start_token=args.start_token,
    )

    neutral_activations = None
    if args.neutral and not args.skip_neutral_pca:
        neutral_records = read_jsonl(args.neutral)
        neutral_texts = [str(record["text"]) for record in neutral_records]
        neutral_activations = mean_hidden_activations(
            model=model,
            tokenizer=tokenizer,
            texts=neutral_texts,
            layer_index=layer_index,
            device=device,
            batch_size=args.batch_size,
            max_length=args.max_length,
            start_token=args.start_token,
        )

    vector_set = build_vector_set(
        story_records=story_records,
        story_activations=story_activations,
        model_id=args.model_id,
        layer_index=layer_index,
        neutral_activations=neutral_activations,
        neutral_variance_threshold=args.neutral_variance_threshold,
    )
    save_vector_set(args.out, vector_set)

    print(f"saved {len(vector_set.emotions)} vectors to {args.out}")
    print(f"model={args.model_id} device={device} dtype={dtype} layer_index={layer_index}")
    print(f"emotions={', '.join(vector_set.emotions)}")


if __name__ == "__main__":
    main()
