from __future__ import annotations

import numpy as np
import torch


def _move_batch(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device) for key, value in batch.items()}


def _effective_start(length: int, start_token: int) -> int:
    if length <= 0:
        return 0
    if length > start_token:
        return start_token
    return max(0, length // 2)


@torch.inference_mode()
def mean_hidden_activations(
    model,
    tokenizer,
    texts: list[str],
    layer_index: int,
    device: torch.device,
    batch_size: int = 4,
    max_length: int = 512,
    start_token: int = 50,
) -> np.ndarray:
    """Return one mean hidden-state vector per text."""
    means: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = _move_batch(encoded, device)
        outputs = model(**encoded, output_hidden_states=True, use_cache=False)
        hidden = outputs.hidden_states[layer_index].detach().float().cpu()
        if not torch.isfinite(hidden).all():
            raise FloatingPointError(
                "Model hidden states contain NaN or Inf values. "
                "Rerun activation-analysis scripts with --device cpu --dtype float32. "
                "Apple Silicon MPS can be unstable for Gemma hidden-state analysis."
            )
        attention = encoded["attention_mask"].detach().cpu()
        for row in range(hidden.shape[0]):
            length = int(attention[row].sum().item())
            begin = _effective_start(length, start_token)
            if begin >= length:
                begin = max(0, length - 1)
            means.append(hidden[row, begin:length].mean(dim=0).numpy())
    return np.stack(means, axis=0)


@torch.inference_mode()
def token_hidden_activations(
    model,
    tokenizer,
    text: str,
    layer_index: int,
    device: torch.device,
    max_length: int = 512,
) -> tuple[list[str], np.ndarray]:
    encoded = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encoded = _move_batch(encoded, device)
    outputs = model(**encoded, output_hidden_states=True, use_cache=False)
    token_ids = encoded["input_ids"][0].detach().cpu().tolist()
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    hidden = outputs.hidden_states[layer_index][0].detach().float().cpu().numpy()
    return tokens, hidden
