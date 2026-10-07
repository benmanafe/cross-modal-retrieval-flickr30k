import io
import os
import random
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel
from transformers import AutoTokenizer

from cmrs.config import DataConfig
from cmrs.data import build_transforms
from cmrs.model import load_checkpoint

DEPLOY = Path(os.environ.get("DEPLOY_DIR", "deploy"))
MAX_UPLOAD = 10 * 1024 * 1024
cfg = DataConfig()
torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))

device = torch.device("cpu")  # CPU is plenty for single-query inference
model = load_checkpoint(DEPLOY / "model.pt", device)
tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer)
_, eval_tf = build_transforms(cfg.img_size)

gallery = torch.load(DEPLOY / "gallery.pt", map_location="cpu", weights_only=False)
filenames, captions, cap_to_img = gallery["filenames"], gallery["captions"], gallery["cap_to_img"]
img_emb, txt_emb = gallery["images"], gallery["texts"]

app = FastAPI(title="Cross-modal retrieval demo")
# Open CORS so a personal website on another domain can call the API directly
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/images", StaticFiles(directory=DEPLOY / "images"), name="images")


class TextQuery(BaseModel):
    query: str
    k: int = 10


def _clamp_k(k):
    return max(1, min(k, 30))


def _image_hits(sims, k):
    scores, idx = sims.topk(k)
    return [{"image": f"/images/{filenames[i]}", "score": round(s, 4)} for i, s in zip(idx.tolist(), scores.tolist())]


def _caption_hits(sims, k, truth_image=None):
    scores, idx = sims.topk(k)
    return [
        {"caption": captions[i], "score": round(s, 4), "correct": truth_image is not None and cap_to_img[i] == truth_image}
        for i, s in zip(idx.tolist(), scores.tolist())
    ]


@torch.no_grad()
def _embed_text(text):
    tok = tokenizer([text], padding=True, truncation=True, max_length=cfg.max_len, return_tensors="pt")
    return model.text_encoder(tok["input_ids"], tok["attention_mask"])


@torch.no_grad()
def _embed_image(image):
    return model.image_encoder(eval_tf(image).unsqueeze(0))


@app.get("/api/info")
def info():
    return {"images": len(filenames), "captions": len(captions), "model": model.cfg.image_backbone}


@app.post("/api/text-to-image")
def text_to_image(body: TextQuery):
    text = body.query.strip()
    if not text:
        raise HTTPException(400, "Query is empty")
    sims = (_embed_text(text) @ img_emb.t()).squeeze(0)
    return {"results": _image_hits(sims, _clamp_k(body.k))}


@app.post("/api/image-to-text")
async def image_to_text(file: UploadFile = File(...), k: int = 10):
    data = await file.read()
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "Image too large (max 10 MB)")
    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Could not read that file as an image")
    sims = (_embed_image(image) @ txt_emb.t()).squeeze(0)
    return {"results": _caption_hits(sims, _clamp_k(k))}


@app.get("/api/random-image")
def random_image(k: int = 10):
    """A random gallery image with its retrieved captions, flagging the ground-truth ones."""
    n = random.randrange(len(filenames))
    sims = (img_emb[n] @ txt_emb.t())
    truth = [c for c, i in zip(captions, cap_to_img) if i == n]
    return {"image": f"/images/{filenames[n]}", "results": _caption_hits(sims, _clamp_k(k), n), "ground_truth": truth}


@app.get("/")
def index():
    return FileResponse("web/index.html")
