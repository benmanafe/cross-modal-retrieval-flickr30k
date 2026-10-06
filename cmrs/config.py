from dataclasses import dataclass, fields
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class ModelConfig:
    embed_dim: int = 512
    image_backbone: str = "efficientnet_b0"
    text_backbone: str = "sentence-transformers/all-MiniLM-L12-v2"
    # Everything before this stage/layer index is frozen (efficientnet: features[k:], MiniLM: layer[k:])
    image_unfreeze_from: int = 6
    text_unfreeze_from: int = 8
    dropout: float = 0.1


@dataclass
class DataConfig:
    split_json: Path = ROOT / "dataset_flickr30k.json"
    image_dir: Path = ROOT / "flickr30k_images"
    # Pre-resized copy of the images (shorter side = cache_size), built once by prepare_data.py
    cache_dir: Path = ROOT / "flickr30k_cache"
    cache_size: int = 256
    img_size: int = 224
    tokenizer: str = "sentence-transformers/all-MiniLM-L12-v2"
    max_len: int = 128
    batch_size: int = 64
    num_workers: int = 4
    seed: int = 42


@dataclass
class TrainConfig:
    run_name: str = "run"
    out_dir: Path = ROOT / "runs"
    epochs: int = 10  # one epoch = one pass over the 29k training images (one random caption each)
    lr_backbone: float = 5e-5
    lr_head: float = 5e-4  # projection heads and logit_scale
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    grad_clip: float = 1.0
    amp: bool = True  # bf16 autocast
    patience: int = 3  # stop after this many epochs without a val rsum improvement


def _build(cls, values):
    values = dict(values or {})
    for f in fields(cls):
        if f.type is Path and f.name in values:
            values[f.name] = Path(values[f.name])
    return cls(**values)


def load_config(path):
    """Reads a YAML file with optional `data`, `model` and `train` sections -> (DataConfig, ModelConfig, TrainConfig)."""
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}
    return _build(DataConfig, raw.get("data")), _build(ModelConfig, raw.get("model")), _build(TrainConfig, raw.get("train"))
