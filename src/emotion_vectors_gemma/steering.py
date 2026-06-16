from __future__ import annotations

from contextlib import contextmanager
from typing import Iterable

import torch

from emotion_vectors_gemma.modeling import find_decoder_layers, transformer_layer_count


def default_steering_layers(model) -> list[int]:
    count = transformer_layer_count(model)
    center = max(0, round(count * 2 / 3) - 1)
    return sorted({max(0, center - 1), center, min(count - 1, center + 1)})


@contextmanager
def steer_residual_stream(
    model,
    direction,
    strength: float,
    layers: Iterable[int] | None = None,
    scale: float | None = None,
):
    """Add a normalized direction to decoder-layer outputs during a context.

    Strength is interpreted as a fraction of the current residual norm when
    scale is not provided, matching the paper's convention approximately.
    """
    modules = find_decoder_layers(model)
    selected_layers = list(default_steering_layers(model) if layers is None else layers)
    handles = []

    def make_hook():
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            vector = torch.as_tensor(direction, device=hidden.device, dtype=hidden.dtype)
            vector = vector / vector.norm().clamp_min(1e-12)
            local_scale = scale
            if local_scale is None:
                local_scale = float(hidden.detach().norm(dim=-1).mean().item())
            delta = vector.view(1, 1, -1) * (strength * local_scale)
            updated = hidden + delta
            if isinstance(output, tuple):
                return (updated,) + output[1:]
            return updated

        return hook

    try:
        for index in selected_layers:
            handles.append(modules[index].register_forward_hook(make_hook()))
        yield
    finally:
        for handle in handles:
            handle.remove()

