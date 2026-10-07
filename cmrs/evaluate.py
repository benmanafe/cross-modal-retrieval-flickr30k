import argparse
import json

import torch
from tqdm import tqdm

from .config import DataConfig
from .data import build_eval_loaders, build_transforms, load_splits
from .model import load_checkpoint


def _autocast(device, amp):
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp and device.type == "cuda")


@torch.no_grad()
def encode_images(model, loader, device, amp=True):
    model.eval()
    out = []
    for images in tqdm(loader, desc="Encoding images", leave=False):
        with _autocast(device, amp):
            out.append(model.image_encoder(images.to(device, non_blocking=True)).float().cpu())
    return torch.cat(out)


@torch.no_grad()
def encode_texts(model, loader, device, amp=True):
    model.eval()
    out = []
    for input_ids, attention_mask in tqdm(loader, desc="Encoding captions", leave=False):
        with _autocast(device, amp):
            emb = model.text_encoder(input_ids.to(device, non_blocking=True), attention_mask.to(device, non_blocking=True))
        out.append(emb.float().cpu())
    return torch.cat(out)


def retrieval_metrics(image_embeds, text_embeds, caption_to_image, ks=(1, 5, 10)):
    """Recall@K and median rank, both directions, for N images and their (usually 5N) captions.

    Image -> text counts a hit if *any* of the image's captions is in the top K.
    Text -> image counts a hit if the caption's image is in the top K.
    """
    sims = image_embeds @ text_embeds.t()  # (N_img, N_txt)
    n_img, n_txt = sims.shape

    # image -> text: rank of the best-ranked ground-truth caption (0-based)
    gt_mask = caption_to_image.unsqueeze(0) == torch.arange(n_img).unsqueeze(1)
    best_gt = sims.masked_fill(~gt_mask, float("-inf")).max(dim=1).values
    i2t_rank = (sims > best_gt.unsqueeze(1)).sum(dim=1)

    # text -> image: rank of the ground-truth image (0-based)
    gt_sim = sims[caption_to_image, torch.arange(n_txt)]
    t2i_rank = (sims > gt_sim.unsqueeze(0)).sum(dim=0)

    metrics = {}
    for name, rank in (("i2t", i2t_rank), ("t2i", t2i_rank)):
        for k in ks:
            metrics[f"{name}_R@{k}"] = (rank < k).float().mean().item() * 100
        metrics[f"{name}_median_rank"] = rank.float().median().item() + 1
    metrics["rsum"] = sum(v for key, v in metrics.items() if "R@" in key)
    return metrics


def evaluate(model, image_loader, text_loader, caption_to_image, device, amp=True):
    image_embeds = encode_images(model, image_loader, device, amp)
    text_embeds = encode_texts(model, text_loader, device, amp)
    return retrieval_metrics(image_embeds, text_embeds, caption_to_image)


def format_metrics(metrics):
    lines = [f"{'':>5} {'R@1':>7} {'R@5':>7} {'R@10':>7} {'MedR':>6}"]
    for name, label in (("i2t", "i->t"), ("t2i", "t->i")):
        lines.append(
            f"{label:>5} {metrics[f'{name}_R@1']:7.1f} {metrics[f'{name}_R@5']:7.1f} "
            f"{metrics[f'{name}_R@10']:7.1f} {metrics[f'{name}_median_rank']:6.0f}"
        )
    lines.append(f"rsum: {metrics['rsum']:.1f}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--out", help="optional path to save the metrics as JSON")
    args = parser.parse_args()

    cfg = DataConfig(batch_size=args.batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_checkpoint(args.checkpoint, device)
    _, eval_tf = build_transforms(cfg.img_size)
    splits = load_splits(cfg.split_json)
    image_loader, text_loader, caption_to_image = build_eval_loaders(cfg, splits[args.split], cfg.cache_dir, eval_tf)

    metrics = evaluate(model, image_loader, text_loader, caption_to_image, device, amp=not args.no_amp)
    print(f"\n{args.checkpoint} on {args.split} ({len(image_loader.dataset)} images, {len(caption_to_image)} captions)")
    print(format_metrics(metrics))

    if args.out:
        with open(args.out, "w") as f:
            json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
