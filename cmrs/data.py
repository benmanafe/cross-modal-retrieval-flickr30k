import json
import random
from multiprocessing import Pool
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from .config import DataConfig

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ------------------------------- Splits -------------------------------
def load_splits(split_json):
    """Returns {'train'|'val'|'test': [(filename, [5 captions]), ...]} using the Karpathy split."""
    with open(split_json, "r") as f:
        images = json.load(f)["images"]

    splits = {"train": [], "val": [], "test": []}
    for img in images:
        captions = [s["raw"].strip() for s in img["sentences"]]
        splits[img["split"]].append((img["filename"], captions))
    return splits


# ------------------------------- Image cache -------------------------------
def _resize_one(args):
    src, dst, size = args
    if dst.exists():
        return
    with Image.open(src) as img:
        img = img.convert("RGB")
        w, h = img.size
        scale = size / min(w, h)
        if scale < 1:  # only ever shrink, never upscale
            img = img.resize((round(w * scale), round(h * scale)), Image.BICUBIC)
        img.save(dst, quality=95)


def build_image_cache(image_dir, cache_dir, size=256, workers=8):
    """Resize every image once so the dataloader doesn't decode full-size JPEGs every epoch."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(p, cache_dir / p.name, size) for p in Path(image_dir).glob("*.jpg")]
    with Pool(workers) as pool:
        for _ in tqdm(pool.imap_unordered(_resize_one, jobs, chunksize=64), total=len(jobs), desc="Caching images"):
            pass


# ------------------------------- Transforms -------------------------------
def build_transforms(img_size=224, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    train = T.Compose([
        T.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    eval_ = T.Compose([
        T.Resize(int(img_size * 256 / 224)),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    return train, eval_


# ------------------------------- Datasets -------------------------------
class TrainDataset(Dataset):
    """One item per *image*; a random one of its 5 captions is drawn each time.

    Because an image appears once per epoch, a batch can never contain the same image twice,
    so the contrastive loss has no false negatives from duplicate images.
    """

    def __init__(self, items, image_dir, transform):
        self.items = items
        self.image_dir = Path(image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        filename, captions = self.items[idx]
        with Image.open(self.image_dir / filename) as img:
            image = self.transform(img.convert("RGB"))
        return image, random.choice(captions)


class ImageDataset(Dataset):
    """Unique images of a split, in split order (used for retrieval evaluation)."""

    def __init__(self, filenames, image_dir, transform):
        self.filenames = filenames
        self.image_dir = Path(image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        with Image.open(self.image_dir / self.filenames[idx]) as img:
            return self.transform(img.convert("RGB"))


class TextDataset(Dataset):
    def __init__(self, captions):
        self.captions = captions

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        return self.captions[idx]


# ------------------------------- Collate & loaders -------------------------------
class TextCollator:
    """Tokenizes a batch, padding only to the longest caption in the batch."""

    def __init__(self, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len

    def _tokenize(self, texts):
        return self.tokenizer(
            list(texts), padding=True, truncation=True, max_length=self.max_len, return_tensors="pt"
        )

    def pair(self, batch):
        images, captions = zip(*batch)
        tok = self._tokenize(captions)
        return torch.stack(images), tok["input_ids"], tok["attention_mask"]

    def text(self, captions):
        tok = self._tokenize(captions)
        return tok["input_ids"], tok["attention_mask"]


def _loader(dataset, cfg, shuffle, collate_fn=None, drop_last=False, num_workers=None, persistent=False):
    # Every worker is a separate process that imports torch; on Windows too many of them
    # exhaust the paging file (WinError 1455), so only the train loader keeps workers alive.
    num_workers = cfg.num_workers if num_workers is None else num_workers
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent and num_workers > 0,
        drop_last=drop_last,
        collate_fn=collate_fn,
    )


def build_train_loader(cfg: DataConfig, splits, image_dir, transform):
    collator = TextCollator(AutoTokenizer.from_pretrained(cfg.tokenizer), cfg.max_len)
    ds = TrainDataset(splits["train"], image_dir, transform)
    # drop_last keeps every batch the same size, which matters for the contrastive loss
    return _loader(ds, cfg, shuffle=True, collate_fn=collator.pair, drop_last=True, persistent=True)


def build_eval_loaders(cfg: DataConfig, items, image_dir, transform):
    """Returns (image_loader, text_loader, caption_to_image) for one split.

    Captions are flattened image by image, so caption k belongs to image caption_to_image[k]
    and image i owns captions caption_to_image == i (normally 5 of them).
    """
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer)
    collator = TextCollator(tokenizer, cfg.max_len)

    filenames = [f for f, _ in items]
    captions, caption_to_image = [], []
    for i, (_, caps) in enumerate(items):
        captions.extend(caps)
        caption_to_image.extend([i] * len(caps))

    image_loader = _loader(
        ImageDataset(filenames, image_dir, transform), cfg, shuffle=False, num_workers=min(2, cfg.num_workers)
    )
    # Tokenizing short captions is cheap, so no worker processes are needed here
    text_loader = _loader(TextDataset(captions), cfg, shuffle=False, collate_fn=collator.text, num_workers=0)
    return image_loader, text_loader, torch.tensor(caption_to_image)


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
