"""Train the cross-modal model.

Usage:
    python -m cmrs.train --config configs/exp1_fixes.yaml
    tensorboard --logdir runs
"""
import argparse
import json
import math
from dataclasses import asdict

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .config import load_config
from .data import build_eval_loaders, build_train_loader, build_transforms, load_splits, seed_everything
from .evaluate import _autocast, evaluate, format_metrics
from .losses import clip_loss
from .model import CrossModalModel, load_checkpoint, save_checkpoint


def build_optimizer(model, tcfg):
    """Four param groups: {backbone, head} x {decay, no-decay}.

    Biases, LayerNorm weights and the scalar logit_scale (all ndim < 2) get no weight decay.
    The pretrained backbones learn slower than the freshly initialised heads.
    """
    groups = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_head = name == "logit_scale" or ".fc." in name
        decay = p.ndim >= 2
        groups.setdefault((is_head, decay), []).append(p)

    param_groups = [
        {
            "params": params,
            "lr": tcfg.lr_head if is_head else tcfg.lr_backbone,
            "weight_decay": tcfg.weight_decay if decay else 0.0,
        }
        for (is_head, decay), params in groups.items()
    ]
    return AdamW(param_groups)


def build_scheduler(optimizer, total_steps, warmup_ratio):
    warmup_steps = max(1, int(total_steps * warmup_ratio))

    def factor(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, factor)


def train_one_epoch(model, loader, optimizer, scheduler, tcfg, device, writer, epoch, global_step, max_steps=None):
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    running = 0.0
    bar = tqdm(loader, desc=f"Epoch {epoch} train", leave=False)

    for i, (images, input_ids, attention_mask) in enumerate(bar):
        if max_steps is not None and i >= max_steps:
            break
        images = images.to(device, non_blocking=True)
        input_ids = input_ids.to(device, non_blocking=True)
        attention_mask = attention_mask.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, tcfg.amp):
            image_embeds, text_embeds = model(images, input_ids, attention_mask)
        loss = clip_loss(image_embeds, text_embeds, model.logit_scale)
        loss.backward()
        clip_grad_norm_(trainable, tcfg.grad_clip)
        optimizer.step()
        scheduler.step()
        model.clamp_logit_scale_()

        running += loss.item()
        global_step += 1
        bar.set_postfix(loss=f"{running / (i + 1):.4f}")
        if global_step % 50 == 0:
            writer.add_scalar("train/loss", loss.item(), global_step)
            writer.add_scalar("train/lr_head", scheduler.get_last_lr()[-1], global_step)
            writer.add_scalar("train/logit_scale", model.logit_scale.exp().item(), global_step)

    return running / max(1, i), global_step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-steps-per-epoch", type=int, help="limit batches per epoch (smoke tests)")
    args = parser.parse_args()

    dcfg, mcfg, tcfg = load_config(args.config)
    run_dir = tcfg.out_dir / tcfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.json", "w") as f:
        json.dump({"data": asdict(dcfg), "model": asdict(mcfg), "train": asdict(tcfg)}, f, indent=2, default=str)

    seed_everything(dcfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_tf, eval_tf = build_transforms(dcfg.img_size)
    splits = load_splits(dcfg.split_json)

    train_loader = build_train_loader(dcfg, splits, dcfg.cache_dir, train_tf)
    val_images, val_texts, val_c2i = build_eval_loaders(dcfg, splits["val"], dcfg.cache_dir, eval_tf)

    model = CrossModalModel(mcfg).to(device)
    optimizer = build_optimizer(model, tcfg)
    steps_per_epoch = min(len(train_loader), args.max_steps_per_epoch or len(train_loader))
    scheduler = build_scheduler(optimizer, tcfg.epochs * steps_per_epoch, tcfg.warmup_ratio)
    writer = SummaryWriter(run_dir / "tensorboard")

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Run '{tcfg.run_name}' on {device} | trainable params: {n_trainable / 1e6:.1f}M | steps/epoch: {steps_per_epoch}")

    best_rsum, bad_epochs, global_step, history = -1.0, 0, 0, []
    for epoch in range(1, tcfg.epochs + 1):
        train_loss, global_step = train_one_epoch(
            model, train_loader, optimizer, scheduler, tcfg, device, writer, epoch, global_step, args.max_steps_per_epoch
        )
        metrics = evaluate(model, val_images, val_texts, val_c2i, device, tcfg.amp)

        writer.add_scalar("epoch/train_loss", train_loss, epoch)
        for key, value in metrics.items():
            writer.add_scalar(f"val/{key}", value, epoch)
        history.append({"epoch": epoch, "train_loss": train_loss, **metrics})
        print(f"Epoch {epoch}/{tcfg.epochs} | train loss {train_loss:.4f} | val rsum {metrics['rsum']:.1f} "
              f"(i->t R@1 {metrics['i2t_R@1']:.1f}, t->i R@1 {metrics['t2i_R@1']:.1f})")

        with open(run_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

        save_checkpoint(run_dir / "last.pt", model, epoch=epoch, metrics=metrics)
        if metrics["rsum"] > best_rsum:
            best_rsum, bad_epochs = metrics["rsum"], 0
            save_checkpoint(run_dir / "best.pt", model, epoch=epoch, metrics=metrics)
        else:
            bad_epochs += 1
            if bad_epochs >= tcfg.patience:
                print(f"No val rsum improvement for {tcfg.patience} epochs, stopping early.")
                break

    # The test split is only touched once, with the checkpoint chosen on validation
    best = load_checkpoint(run_dir / "best.pt", device)
    test_images, test_texts, test_c2i = build_eval_loaders(dcfg, splits["test"], dcfg.cache_dir, eval_tf)
    test_metrics = evaluate(best, test_images, test_texts, test_c2i, device, tcfg.amp)
    with open(run_dir / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)
    print(f"\nBest checkpoint (val rsum {best_rsum:.1f}) on test:\n{format_metrics(test_metrics)}")
    writer.close()


if __name__ == "__main__":
    main()
