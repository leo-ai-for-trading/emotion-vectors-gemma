from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


EMOTIONAL_STORY_PROMPT = """Write {n_stories} different short stories based on the following premise.

Topic: {topic}

The story should follow a character who is feeling {emotion}.
The topic event must be central to each story, not just background detail.

Format each story with a line containing only <new story> before it.

The stories should be fresh starts with no continuity. Use a mix of first-person
and third-person narration.

Each story should be one paragraph, roughly 100-140 words.

Begin immediately with <new story>. Do not include any introduction, summary, or
commentary before the first story.

Important: do not use the word "{emotion}" in the stories. Also avoid these
nearby labels: {forbidden_terms}.

Convey the emotion through actions, physical sensations, body language,
dialogue, tone, thoughts, and situational context.
"""


FORBIDDEN_EMOTION_TERMS = {
    "afraid": ["afraid", "fear", "fearful", "scared", "terrified"],
    "angry": ["angry", "anger", "furious", "mad", "rage"],
    "ashamed": ["ashamed", "shame", "shameful", "embarrassed", "humiliated"],
    "calm": ["calm", "peaceful", "serene", "relaxed", "tranquil"],
    "curious": ["curious", "curiosity", "inquisitive", "intrigued"],
    "desperate": ["desperate", "desperation", "frantic", "hopeless"],
    "envious": ["envious", "envy", "jealous", "jealousy"],
    "frustrated": ["frustrated", "frustration", "annoyed", "exasperated"],
    "guilty": ["guilty", "guilt", "remorse", "remorseful"],
    "happy": ["happy", "happiness", "joy", "joyful", "cheerful", "glad"],
    "hopeful": ["hopeful", "hope", "optimistic", "expectant"],
    "loving": ["loving", "love", "affection", "adoring", "tender"],
    "nervous": ["nervous", "anxious", "anxiety", "uneasy", "jittery"],
    "proud": ["proud", "pride", "triumphant", "accomplished"],
    "relieved": ["relieved", "relief", "reassured"],
    "sad": ["sad", "sadness", "sorrow", "grief", "melancholy"],
}


def forbidden_terms_for(emotion: str) -> str:
    terms = FORBIDDEN_EMOTION_TERMS.get(emotion.lower(), [emotion])
    return ", ".join(dict.fromkeys(terms))


def read_jsonl(path: str | Path) -> list[dict]:
    records: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_number}") from exc
    return records


def write_jsonl(path: str | Path, records: Iterable[dict]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_list(path: str | Path) -> list[str]:
    values: list[str] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                values.append(stripped)
    return values


def story_texts(records: list[dict]) -> list[str]:
    missing = [index for index, record in enumerate(records) if "text" not in record]
    if missing:
        raise ValueError(f"Story records missing 'text' field at indices {missing[:5]}")
    return [str(record["text"]) for record in records]


def validate_story_records(records: list[dict]) -> None:
    required = {"emotion", "text"}
    for index, record in enumerate(records):
        missing = required - set(record)
        if missing:
            raise ValueError(f"Story record {index} is missing fields: {sorted(missing)}")
