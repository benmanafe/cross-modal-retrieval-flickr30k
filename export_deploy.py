"""Builds the self-contained `deploy/` folder the web demo needs.

    python export_deploy.py [--checkpoint runs/exp3_convnext/best.pt] [--split test]

deploy/
    model.pt      weights + model config (no optimizer state)
    gallery.pt    filenames, captions, caption->image map, image/text embeddings
    images/       the gallery images (256px cache copies)
"""
import argparse
import shutil
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from cmrs.config import DataConfig
from cmrs.data import ImageDataset, TextCollator, TextDataset, build_transforms, load_splits
from cmrs.evaluate import encode_images, encode_texts
from cmrs.model import load_checkpoint, save_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="runs/exp3_convnext/best.pt")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--out", default="deploy")
    args = parser.parse_args()

    cfg = DataConfig()
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_checkpoint(args.checkpoint, device)
    save_checkpoint(out / "model.pt", model)

    items = load_splits(cfg.split_json)[args.split]
    filenames = [f for f, _ in items]
    captions, cap_to_img = [], []
    for i, (_, caps) in enumerate(items):
        captions.extend(caps)
        cap_to_img.extend([i] * len(caps))

    _, eval_tf = build_transforms(cfg.img_size)
    collator = TextCollator(AutoTokenizer.from_pretrained(cfg.tokenizer), cfg.max_len)
    img_emb = encode_images(model, DataLoader(ImageDataset(filenames, cfg.cache_dir, eval_tf), batch_size=128), device)
    txt_emb = encode_texts(model, DataLoader(TextDataset(captions), batch_size=256, collate_fn=collator.text), device)

    torch.save(
        {"filenames": filenames, "captions": captions, "cap_to_img": cap_to_img, "images": img_emb, "texts": txt_emb},
        out / "gallery.pt",
    )
    for name in filenames:
        shutil.copy(cfg.cache_dir / name, out / "images" / name)

    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 1e6
    print(f"Wrote {out}/ ({len(filenames)} images, {len(captions)} captions, {size:.0f} MB)")


if __name__ == "__main__":
    main()
