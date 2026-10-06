# Cross-Modal Retrieval on Flickr30ka

Search images with a sentence, or find captions for an image. A dual-encoder model (image encoder + text encoder) is trained contrastively on Flickr30k so that matching images and captions land close together in a shared embedding space.

The repo includes the training code, the experiments that led to the final model, and a small web demo.

## Results

Evaluated on the Karpathy 1K test split (1,000 images, 5,000 captions). Image→text counts a hit if any of the image's five captions is in the top K.

| Run | Image backbone | Embed dim | Test rsum | i→t R@1 | t→i R@1 |
|---|---|---|---|---|---|
| Baseline (10 epochs) | EfficientNet-B0 | 1024 | 294.1 | 29.6 | 22.9 |
| exp2_unfreeze | EfficientNet-B0 | 512 | 388.6 | 46.6 | 36.7 |
| exp1_fixes | EfficientNet-B0 | 1024 | 402.7 | 51.4 | 38.0 |
| **exp3_convnext** | **ConvNeXt-Tiny** | 1024 | **447.7** | **58.0** | **47.3** |

Full result for the best model (checkpoint chosen by validation rsum, 440.9):

|  | R@1 | R@5 | R@10 | MedR |
|---|---|---|---|---|
| Image → Text | 58.0 | 85.9 | 92.4 | 1 |
| Text → Image | 47.3 | 77.9 | 86.3 | 2 |

**What the experiments showed**
- Swapping EfficientNet-B0 for ConvNeXt-Tiny, with every other setting identical to exp1, gave about +45 rsum. This was the biggest single improvement.
- exp1 and exp2 differ in several settings at once (embed dim, unfreeze depth, backbone learning rate), so the 14-point gap between them can't be attributed to one factor.
- exp3's validation score is flat from about epoch 16 while training loss keeps falling, so longer training is unlikely to help much.
- Validation and test scores agree to within about 10 rsum in every run.

## Model

- **Image encoder:** ConvNeXt-Tiny (ImageNet-pretrained), global average pool, LayerNorm, linear projection to 1024 dimensions. The stem and the first two stages stay frozen.
- **Text encoder:** `sentence-transformers/all-MiniLM-L12-v2` with masked mean pooling and a linear projection to 1024 dimensions. The first 8 of 12 layers stay frozen.
- **Loss:** symmetric InfoNCE (CLIP-style) with a learnable temperature.
- **Training:** AdamW, separate learning rates for backbones (5e-5) and heads (5e-4), warmup, gradient clipping, bf16 autocast, early stopping on validation rsum. Each epoch draws one random caption per training image, so a batch never contains the same image twice.

## Project layout

```
cmrs/                 training and evaluation package
  model.py            image/text encoders, checkpoint save/load
  losses.py           contrastive loss
  data.py             Karpathy splits, datasets, transforms, loaders
  train.py            training loop
  evaluate.py         Recall@K / median rank
  prepare_data.py     builds the resized image cache
configs/              one YAML per experiment
server.py             FastAPI backend for the demo
web/index.html        demo front end
export_deploy.py      packages the model, embeddings and images for the demo
run_demo.bat          Windows launcher for the demo
```

## Setup

Python 3.10+ and a CUDA GPU are recommended for training. For GPU training, install the CUDA build of PyTorch from [pytorch.org](https://pytorch.org) first, then:

```bash
pip install -r requirements.txt
```

Dataset: download [Flickr30k](https://shannon.cs.illinois.edu/DenotationGraph/) images into `flickr30k_images/` and Karpathy's `dataset_flickr30k.json` into the project root. Flickr30k is licensed for non-commercial research use, so it is not included in this repo.

## Usage

```bash
# 1. Resize images once so the dataloader isn't decoding full-size JPEGs every epoch
python -m cmrs.prepare_data

# 2. Train (writes best.pt, last.pt, history.json and TensorBoard logs to runs/<run_name>/)
python -m cmrs.train --config configs/exp3_convnext.yaml

# 3. Evaluate a checkpoint
python -m cmrs.evaluate --checkpoint runs/exp3_convnext/best.pt --split test
```

## Web demo

The demo searches the 1,000 test images (which the model never trained on) and their 5,000 captions.

```bash
# Package the trained model into deploy/ (model weights, precomputed embeddings, gallery images)
python export_deploy.py --checkpoint runs/exp3_convnext/best.pt

# Start the server, then open http://localhost:8000
uvicorn server:app --port 8000
```

On Windows, `run_demo.bat` starts it using the `deep_learning` conda environment. Edit the environment name in that file if yours differs.

- **Text → Image:** type a description and get the closest images.
- **Image → Text:** upload a photo or pick a random gallery image. For gallery images, ground-truth captions are marked.
- **API:** `POST /api/text-to-image`, `POST /api/image-to-text`, `GET /api/random-image`, `GET /api/info`. CORS is open, so another site can call it.

A text query takes about 60 ms on CPU. To share a demo from your own machine, run the server and then `cloudflared tunnel --url http://localhost:8000` for a temporary public link.

### Limitations
- It is retrieval, not captioning. Results come from the fixed gallery, so an uploaded photo is matched to the nearest existing caption.
- Flickr30k is mostly people and everyday scenes, so abstract or unusual queries return the nearest scene of that kind.

## Model weights

Checkpoints are about 250 MB and are not in the repo (GitHub rejects files over 100 MB). Train your own with the commands above, or download the weights from the releases page if provided.

## Acknowledgements

- Data: [Flickr30k](https://shannon.cs.illinois.edu/DenotationGraph/) with the [Karpathy splits](https://cs.stanford.edu/people/karpathy/deepimagesent/).
- Backbones: torchvision ConvNeXt / EfficientNet and `sentence-transformers/all-MiniLM-L12-v2`.
