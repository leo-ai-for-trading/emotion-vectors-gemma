#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from collections import Counter

from emotion_vectors_gemma.datasets import read_jsonl, read_list


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate generated story dataset balance.")
    parser.add_argument("--stories", required=True)
    parser.add_argument("--emotions")
    parser.add_argument("--topics")
    parser.add_argument("--expected-per-cell", type=int)
    parser.add_argument("--min-words", type=int, default=35)
    return parser.parse_args()


def contains_emotion_word(text: str, emotion: str) -> bool:
    escaped = re.escape(emotion)
    pattern = rf"(?<![A-Za-z]){escaped}(?![A-Za-z])"
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def main() -> None:
    args = parse_args()
    records = read_jsonl(args.stories)
    emotions = read_list(args.emotions) if args.emotions else sorted({r["emotion"] for r in records})
    topics = read_list(args.topics) if args.topics else sorted({r["topic"] for r in records})

    by_emotion = Counter(str(record["emotion"]) for record in records)
    by_cell = Counter((str(record["emotion"]), str(record["topic"])) for record in records)
    short_records = [
        index
        for index, record in enumerate(records)
        if len(str(record["text"]).split()) < args.min_words
    ]
    explicit_emotion = [
        index
        for index, record in enumerate(records)
        if contains_emotion_word(str(record["text"]), str(record["emotion"]))
    ]
    missing_cells = [
        (emotion, topic)
        for emotion in emotions
        for topic in topics
        if by_cell[(emotion, topic)] == 0
    ]
    underfilled_cells = []
    if args.expected_per_cell is not None:
        underfilled_cells = [
            (emotion, topic, by_cell[(emotion, topic)])
            for emotion in emotions
            for topic in topics
            if by_cell[(emotion, topic)] < args.expected_per_cell
        ]

    print(f"records: {len(records)}")
    print(f"emotions: {len(emotions)}")
    for emotion in emotions:
        print(f"  {emotion}: {by_emotion[emotion]}")
    print(f"topics: {len(topics)}")
    print(f"missing cells: {len(missing_cells)}")
    print(f"underfilled cells: {len(underfilled_cells)}")
    print(f"short records < {args.min_words} words: {len(short_records)}")
    print(f"records containing exact emotion word: {len(explicit_emotion)}")

    if missing_cells[:10]:
        print("first missing cells:")
        for emotion, topic in missing_cells[:10]:
            print(f"  {emotion} / {topic}")
    if underfilled_cells[:10]:
        print("first underfilled cells:")
        for emotion, topic, count in underfilled_cells[:10]:
            print(f"  {emotion} / {topic}: {count}")
    if short_records[:10]:
        print(f"first short record indices: {short_records[:10]}")
    if explicit_emotion[:10]:
        print(f"first exact-emotion indices: {explicit_emotion[:10]}")

    failed = bool(missing_cells or underfilled_cells or short_records or explicit_emotion)
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
