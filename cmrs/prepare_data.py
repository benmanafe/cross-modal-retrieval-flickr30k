"""One-off: build the pre-resized image cache.   Usage: python -m cmrs.prepare_data"""
from .config import DataConfig
from .data import build_image_cache

if __name__ == "__main__":
    cfg = DataConfig()
    build_image_cache(cfg.image_dir, cfg.cache_dir, cfg.cache_size)
