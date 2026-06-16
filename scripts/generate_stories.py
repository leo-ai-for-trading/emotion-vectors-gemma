#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

import torch

from emotion_vectors_gemma.datasets import (
    EMOTIONAL_STORY_PROMPT,
    forbidden_terms_for,
    read_list,
    write_jsonl,
)
from emotion_vectors_gemma.modeling import load_model, load_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic emotional stories.")
    parser.add_argument("--model-id", default="google/gemma-3-1b-it")
    parser.add_argument("--emotions", required=True)
    parser.add_argument("--topics", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--raw-out")
    parser.add_argument("--n-per-prompt", type=int, default=2)
    parser.add_argument("--min-stories-per-prompt", type=int)
    parser.add_argument("--min-story-words", type=int, default=45)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--limit-emotions", type=int)
    parser.add_argument("--limit-topics", type=int)
    parser.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=700)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    return parser.parse_args()


def prompt_to_model_input(tokenizer, prompt: str) -> str:
    if getattr(tokenizer, "chat_template", None):
        try:
            messages = [{"role": "user", "content": prompt}]
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except ImportError as exc:
            if "jinja2" not in str(exc).lower():
                raise
            return gemma_chat_prompt(tokenizer, prompt)
    return prompt


def gemma_chat_prompt(tokenizer, prompt: str) -> str:
    bos = tokenizer.bos_token or "<bos>"
    return (
        f"{bos}<start_of_turn>user\n"
        f"{prompt.strip()}<end_of_turn>\n"
        "<start_of_turn>model\n"
    )


def split_stories(text: str) -> list[str]:
    cleaned = text.replace("\r\n", "\n").strip()
    cleaned = re.sub(r"```(?:text)?", "", cleaned, flags=re.IGNORECASE)
    parts = re.split(
        r"(?:^|\n)\s*(?:<new story>|story\s+\d+\s*:?)\s*(?:\n|$)",
        cleaned,
        flags=re.IGNORECASE,
    )
    stories = [clean_story(part) for part in parts if len(part.strip()) > 40]
    if len(stories) <= 1:
        stories = [
            clean_story(part)
            for part in re.split(r"\n\s*\d+[\).]\s+", cleaned)
            if len(part.strip()) > 40
        ]
    unique: list[str] = []
    seen = set()
    for story in stories:
        key = re.sub(r"\s+", " ", story).lower()
        if story and key not in seen and not looks_like_prompt_artifact(story):
            unique.append(story)
            seen.add(key)
    return unique


def clean_story(text: str) -> str:
    story = text.strip()
    story = re.sub(r"^\s*[-*]\s+", "", story)
    story = re.sub(r"^\s*[\"']", "", story)
    story = re.sub(r"[\"']\s*$", "", story)
    story = re.sub(r"\n{3,}", "\n\n", story)
    story = trim_trailing_junk(story)
    return story.strip()


def trim_trailing_junk(text: str) -> str:
    stripped = text.rstrip()
    tail_start = max(0, len(stripped) - 120)
    for index in range(tail_start, len(stripped)):
        if not is_safe_story_terminal(stripped[index]):
            return stripped[:index].rstrip()
    return stripped


def is_safe_story_terminal(char: str) -> bool:
    if char.isascii():
        return True
    if char in {"’", "‘", "“", "”", "–", "—", "…"}:
        return True
    name = unicodedata.name(char, "")
    return "LATIN" in name


def looks_like_prompt_artifact(text: str) -> bool:
    lowered = text.lower()
    artifacts = [
        "okay, here are",
        "here are two",
        "here are ",
        "format each story",
        "the story should follow",
        "important: do not use",
        "following your specifications",
        "avoiding the word",
        "topic:",
        "write ",
    ]
    return any(artifact in lowered[:300] for artifact in artifacts)


def contains_emotion_word(text: str, emotion: str) -> bool:
    escaped = re.escape(emotion)
    pattern = rf"(?<![A-Za-z]){escaped}(?![A-Za-z])"
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def story_is_valid(text: str, emotion: str, min_words: int) -> bool:
    if len(text.split()) < min_words:
        return False
    if looks_like_prompt_artifact(text):
        return False
    if contains_emotion_word(text, emotion):
        return False
    if has_repeated_junk(text):
        return False
    return True


def has_repeated_junk(text: str) -> bool:
    if re.search(r"(.)\1{8,}", text):
        return True
    non_ascii = sum(1 for char in text if ord(char) > 127)
    return non_ascii > max(20, len(text) * 0.15)


def main() -> None:
    args = parse_args()
    emotions = read_list(args.emotions)
    topics = read_list(args.topics)
    if args.limit_emotions:
        emotions = emotions[: args.limit_emotions]
    if args.limit_topics:
        topics = topics[: args.limit_topics]
    min_stories = args.min_stories_per_prompt or args.n_per_prompt

    tokenizer = load_tokenizer(args.model_id)
    model, device, _dtype = load_model(args.model_id, device=args.device, dtype=args.dtype)
    records = []
    raw_records = []
    raw_out = Path(args.raw_out) if args.raw_out else None

    for emotion in emotions:
        for topic in topics:
            accepted: list[str] = []
            for attempt in range(args.max_retries + 1):
                prompt = EMOTIONAL_STORY_PROMPT.format(
                    n_stories=args.n_per_prompt,
                    topic=topic,
                    emotion=emotion,
                    forbidden_terms=forbidden_terms_for(emotion),
                )
                if attempt:
                    prompt += (
                        "\n\nThe previous attempt did not produce enough clearly "
                        "separated stories. Make sure to include exactly "
                        f"{args.n_per_prompt} sections, each preceded by <new story>."
                    )
                model_input = prompt_to_model_input(tokenizer, prompt)
                encoded = tokenizer(model_input, return_tensors="pt").to(device)
                with torch.inference_mode():
                    generated = model.generate(
                        **encoded,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=True,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        repetition_penalty=args.repetition_penalty,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                continuation = generated[0, encoded["input_ids"].shape[-1] :]
                text = tokenizer.decode(continuation, skip_special_tokens=True)
                stories = split_stories(text)
                raw_records.append(
                    {
                        "emotion": emotion,
                        "topic": topic,
                        "attempt": attempt,
                        "parsed_count": len(stories),
                        "raw_text": text,
                    }
                )
                if raw_out:
                    write_jsonl(raw_out, raw_records)
                for story in stories:
                    if not story_is_valid(story, emotion, args.min_story_words):
                        continue
                    if len(accepted) < args.n_per_prompt:
                        accepted.append(story)
                if len(accepted) >= min_stories:
                    break

            for index, story in enumerate(accepted[: args.n_per_prompt]):
                records.append(
                    {
                        "emotion": emotion,
                        "topic": topic,
                        "story_index": index,
                        "text": story,
                    }
                )
            write_jsonl(args.out, records)
            print(
                f"{emotion} / {topic}: accepted={len(accepted[: args.n_per_prompt])} "
                f"total stories={len(records)}"
            )

    print(f"wrote {len(records)} stories to {args.out}")


if __name__ == "__main__":
    main()
