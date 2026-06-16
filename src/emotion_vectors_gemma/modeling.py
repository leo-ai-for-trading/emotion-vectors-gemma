from __future__ import annotations

import os
from typing import Literal

import torch
from huggingface_hub import get_token
from transformers import AutoModelForCausalLM, AutoTokenizer


DeviceName = Literal["auto", "cpu", "cuda", "mps"]


def pick_device(device: DeviceName = "auto") -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pick_dtype(device: torch.device, dtype: str = "auto") -> torch.dtype:
    if dtype != "auto":
        if not hasattr(torch, dtype):
            raise ValueError(f"Unknown torch dtype: {dtype}")
        selected = getattr(torch, dtype)
        if not isinstance(selected, torch.dtype):
            raise ValueError(f"Not a torch dtype: {dtype}")
        return selected
    if device.type == "cuda":
        return torch.bfloat16
    if device.type == "mps":
        return torch.float32
    return torch.float32


def load_tokenizer(model_id: str, token: str | None = None):
    auth_token = resolve_hf_token(token)
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=auth_token)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(
    model_id: str,
    device: DeviceName = "auto",
    dtype: str = "auto",
    token: str | None = None,
):
    selected_device = pick_device(device)
    selected_dtype = pick_dtype(selected_device, dtype)
    auth_token = resolve_hf_token(token)
    kwargs = {
        "token": auth_token,
        "low_cpu_mem_usage": True,
    }
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=selected_dtype,
            **kwargs,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=selected_dtype,
            **kwargs,
        )
    model.to(selected_device)
    model.eval()
    return model, selected_device, selected_dtype


def resolve_hf_token(token: str | None = None) -> str | None:
    return (
        token
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or get_token()
    )


def transformer_layer_count(model) -> int:
    config = getattr(model, "config", None)
    for attr in ("num_hidden_layers", "n_layer", "num_layers"):
        value = getattr(config, attr, None)
        if value is not None:
            return int(value)
    layers = find_decoder_layers(model)
    return len(layers)


def resolve_hidden_state_index(model, layer: str | int | None) -> int:
    layer_count = transformer_layer_count(model)
    if layer is None or layer == "auto":
        return max(1, round(layer_count * 2 / 3))
    index = int(layer)
    if index < 0:
        index = layer_count + 1 + index
    if index < 0 or index > layer_count:
        raise ValueError(
            f"Layer index {index} is outside hidden-state range 0..{layer_count}"
        )
    return index


def find_decoder_layers(model):
    candidates = [
        ("model", "layers"),
        ("language_model", "model", "layers"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
    ]
    for path in candidates:
        value = model
        for attr in path:
            value = getattr(value, attr, None)
            if value is None:
                break
        if value is not None:
            return list(value)
    raise ValueError("Could not find decoder layers for this model architecture")
