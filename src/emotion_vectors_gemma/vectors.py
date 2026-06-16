from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA


@dataclass(frozen=True)
class EmotionVectorSet:
    model_id: str
    layer_index: int
    emotions: list[str]
    vectors: np.ndarray
    global_mean: np.ndarray
    neutral_components: np.ndarray
    metadata: dict

    def normalized(self) -> np.ndarray:
        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        return self.vectors / norms

    def emotion_index(self, emotion: str) -> int:
        try:
            return self.emotions.index(emotion)
        except ValueError as exc:
            raise KeyError(f"Unknown emotion '{emotion}'. Available: {self.emotions}") from exc

    def vector_for(self, emotion: str, normalize: bool = True) -> np.ndarray:
        vectors = self.normalized() if normalize else self.vectors
        return vectors[self.emotion_index(emotion)]


def project_out_components(vectors: np.ndarray, components: np.ndarray) -> np.ndarray:
    if components.size == 0:
        return vectors
    return vectors - (vectors @ components.T) @ components


def neutral_pca_components(
    neutral_activations: np.ndarray | None,
    variance_threshold: float = 0.5,
) -> np.ndarray:
    if neutral_activations is None or len(neutral_activations) < 2:
        width = 0 if neutral_activations is None else neutral_activations.shape[1]
        return np.zeros((0, width), dtype=np.float32)
    pca = PCA(n_components=variance_threshold, svd_solver="full")
    pca.fit(neutral_activations.astype(np.float32))
    return pca.components_.astype(np.float32)


def build_vector_set(
    story_records: list[dict],
    story_activations: np.ndarray,
    model_id: str,
    layer_index: int,
    neutral_activations: np.ndarray | None = None,
    neutral_variance_threshold: float = 0.5,
) -> EmotionVectorSet:
    if len(story_records) != len(story_activations):
        raise ValueError("story_records and story_activations must have the same length")

    emotions = sorted({str(record["emotion"]) for record in story_records})
    global_mean = story_activations.mean(axis=0).astype(np.float32)
    vectors = []
    counts = {}
    for emotion in emotions:
        indices = [
            index
            for index, record in enumerate(story_records)
            if str(record["emotion"]) == emotion
        ]
        counts[emotion] = len(indices)
        emotion_mean = story_activations[indices].mean(axis=0)
        vectors.append(emotion_mean - global_mean)
    vector_array = np.stack(vectors, axis=0).astype(np.float32)

    components = neutral_pca_components(neutral_activations, neutral_variance_threshold)
    if components.size:
        vector_array = project_out_components(vector_array, components).astype(np.float32)

    return EmotionVectorSet(
        model_id=model_id,
        layer_index=layer_index,
        emotions=emotions,
        vectors=vector_array,
        global_mean=global_mean,
        neutral_components=components,
        metadata={
            "story_count": len(story_records),
            "counts_by_emotion": counts,
            "neutral_count": 0 if neutral_activations is None else int(len(neutral_activations)),
            "neutral_variance_threshold": neutral_variance_threshold,
        },
    )


def save_vector_set(path: str | Path, vector_set: EmotionVectorSet) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        **vector_set.metadata,
        "model_id": vector_set.model_id,
        "layer_index": vector_set.layer_index,
    }
    np.savez_compressed(
        output,
        emotions=np.array(vector_set.emotions),
        vectors=vector_set.vectors,
        global_mean=vector_set.global_mean,
        neutral_components=vector_set.neutral_components,
        metadata=json.dumps(metadata),
    )


def load_vector_set(path: str | Path) -> EmotionVectorSet:
    data = np.load(Path(path), allow_pickle=False)
    metadata = json.loads(str(data["metadata"]))
    return EmotionVectorSet(
        model_id=str(metadata["model_id"]),
        layer_index=int(metadata["layer_index"]),
        emotions=[str(value) for value in data["emotions"].tolist()],
        vectors=data["vectors"].astype(np.float32),
        global_mean=data["global_mean"].astype(np.float32),
        neutral_components=data["neutral_components"].astype(np.float32),
        metadata=metadata,
    )


def mean_probe_scores(hidden: np.ndarray, directions: np.ndarray, start_token: int = 0) -> np.ndarray:
    begin = min(start_token, max(0, hidden.shape[0] - 1))
    return hidden[begin:] @ directions.T

