#!/usr/bin/env python3
from __future__ import annotations

import argparse

from emotion_vectors_gemma.activations import token_hidden_activations
from emotion_vectors_gemma.datasets import read_jsonl, write_jsonl
from emotion_vectors_gemma.modeling import load_model, load_tokenizer
from emotion_vectors_gemma.vectors import load_vector_set, mean_probe_scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score texts with saved emotion vectors.")
    parser.add_argument("--model-id")
    parser.add_argument("--vectors", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--emotions", help="Comma-separated subset of emotions to score.")
    parser.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--start-token", type=int, default=0)
    parser.add_argument("--include-token-scores", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    vector_set = load_vector_set(args.vectors)
    model_id = args.model_id or vector_set.model_id
    tokenizer = load_tokenizer(model_id)
    model, device, _dtype = load_model(model_id, device=args.device, dtype=args.dtype)

    if args.emotions:
        emotions = [value.strip() for value in args.emotions.split(",") if value.strip()]
        indices = [vector_set.emotion_index(emotion) for emotion in emotions]
    else:
        emotions = vector_set.emotions
        indices = list(range(len(emotions)))
    directions = vector_set.normalized()[indices]

    outputs = []
    for record in read_jsonl(args.input):
        text = str(record["text"])
        tokens, hidden = token_hidden_activations(
            model=model,
            tokenizer=tokenizer,
            text=text,
            layer_index=vector_set.layer_index,
            device=device,
            max_length=args.max_length,
        )
        scores = mean_probe_scores(hidden, directions, start_token=args.start_token)
        mean_scores = scores.mean(axis=0)
        output = {
            **record,
            "scores": {
                emotion: float(score) for emotion, score in zip(emotions, mean_scores)
            },
        }
        if args.include_token_scores:
            output["tokens"] = tokens
            output["token_scores"] = [
                {
                    emotion: float(score)
                    for emotion, score in zip(emotions, row)
                }
                for row in scores
            ]
        outputs.append(output)

    write_jsonl(args.output, outputs)
    print(f"wrote {len(outputs)} scored texts to {args.output}")


if __name__ == "__main__":
    main()
