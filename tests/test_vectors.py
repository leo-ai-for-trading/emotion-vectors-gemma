from __future__ import annotations

import numpy as np

from emotion_vectors_gemma.vectors import (
    build_vector_set,
    project_out_components,
)


def test_build_vector_set_subtracts_global_mean() -> None:
    records = [
        {"emotion": "happy", "text": "a"},
        {"emotion": "happy", "text": "b"},
        {"emotion": "sad", "text": "c"},
        {"emotion": "sad", "text": "d"},
    ]
    activations = np.array(
        [
            [2.0, 1.0],
            [2.0, 3.0],
            [-2.0, 1.0],
            [-2.0, -1.0],
        ],
        dtype=np.float32,
    )
    vector_set = build_vector_set(
        story_records=records,
        story_activations=activations,
        model_id="test-model",
        layer_index=3,
    )

    assert vector_set.emotions == ["happy", "sad"]
    np.testing.assert_allclose(vector_set.global_mean, [0.0, 1.0])
    np.testing.assert_allclose(vector_set.vectors[0], [2.0, 1.0])
    np.testing.assert_allclose(vector_set.vectors[1], [-2.0, -1.0])


def test_project_out_components_removes_component_direction() -> None:
    vectors = np.array([[2.0, 3.0], [-4.0, 5.0]], dtype=np.float32)
    components = np.array([[1.0, 0.0]], dtype=np.float32)
    projected = project_out_components(vectors, components)
    np.testing.assert_allclose(projected, [[0.0, 3.0], [0.0, 5.0]])

