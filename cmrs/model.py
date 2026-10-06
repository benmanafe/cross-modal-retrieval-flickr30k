from dataclasses import asdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from transformers import AutoConfig, AutoModel

from .config import ModelConfig


def _build_image_backbone(name, pretrained):
    """Returns (stages, norm, feat_dim).

    `stages` is an nn.Sequential producing a (B, C, H, W) map; `image_unfreeze_from` indexes into it.
    `norm` is the pretrained pooled-feature norm (ConvNeXt's final LayerNorm), Identity otherwise.
    """
    if name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        return models.efficientnet_b0(weights=weights).features, nn.Identity(), 1280
    if name == "convnext_tiny":
        # features = [stem, stage1, down, stage2, down, stage3, down, stage4]
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        net = models.convnext_tiny(weights=weights)
        return net.features, net.classifier[0], 768
    raise ValueError(f"Unsupported image backbone: {name}")


class ImageEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig, pretrained=True):
        super().__init__()
        self.backbone, self.norm, feat_dim = _build_image_backbone(cfg.image_backbone, pretrained)
        self.unfreeze_from = cfg.image_unfreeze_from

        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.backbone[self.unfreeze_from:].parameters():
            p.requires_grad = True

        self.dropout = nn.Dropout(cfg.dropout)
        self.fc = nn.Linear(feat_dim, cfg.embed_dim)

    def frozen_modules(self):
        return list(self.backbone[: self.unfreeze_from])

    def forward(self, images):
        features = self.backbone(images).mean(dim=(2, 3), keepdim=True)  # global average pool -> (B, C, 1, 1)
        features = self.norm(features).flatten(1)
        embeddings = self.fc(self.dropout(features))
        return F.normalize(embeddings, dim=-1)


class TextEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig, pretrained=True):
        super().__init__()
        if pretrained:
            self.transformer = AutoModel.from_pretrained(cfg.text_backbone)
        else:  # architecture only, weights come from a checkpoint
            self.transformer = AutoModel.from_config(AutoConfig.from_pretrained(cfg.text_backbone))
        self.unfreeze_from = cfg.text_unfreeze_from

        for p in self.transformer.parameters():
            p.requires_grad = False
        for p in self.transformer.encoder.layer[self.unfreeze_from:].parameters():
            p.requires_grad = True

        self.dropout = nn.Dropout(cfg.dropout)
        self.fc = nn.Linear(self.transformer.config.hidden_size, cfg.embed_dim)

    def frozen_modules(self):
        return [self.transformer.embeddings, *self.transformer.encoder.layer[: self.unfreeze_from]]

    def forward(self, input_ids, attention_mask):
        token_embeddings = self.transformer(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

        # Masked mean pooling over the real (non-padding) tokens
        mask = attention_mask.unsqueeze(-1).to(token_embeddings.dtype)
        pooled = (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

        embeddings = self.fc(self.dropout(pooled))
        return F.normalize(embeddings, dim=-1)


class CrossModalModel(nn.Module):
    def __init__(self, cfg: ModelConfig, pretrained=True):
        super().__init__()
        self.cfg = cfg
        self.image_encoder = ImageEncoder(cfg, pretrained)
        self.text_encoder = TextEncoder(cfg, pretrained)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def train(self, mode=True):
        super().train(mode)
        if mode:
            # Frozen layers stay in eval mode so their BatchNorm statistics and dropout don't change
            for encoder in (self.image_encoder, self.text_encoder):
                for module in encoder.frozen_modules():
                    module.eval()
        return self

    @torch.no_grad()
    def clamp_logit_scale_(self, max_scale=100.0):
        self.logit_scale.clamp_(0, np.log(max_scale))

    def forward(self, images, input_ids, attention_mask):
        return self.image_encoder(images), self.text_encoder(input_ids, attention_mask)


# ------------------------------- Checkpoints -------------------------------
def save_checkpoint(path, model: CrossModalModel, **extra):
    torch.save({"model": model.state_dict(), "model_cfg": asdict(model.cfg), **extra}, path)


def _remap_legacy_keys(state_dict):
    """Maps the first-version checkpoints (raw state_dict, `image_encoder.resnet.0.*`) to the current names."""
    remapped = {}
    for key, value in state_dict.items():
        if key.endswith("position_ids"):  # non-persistent buffer in recent transformers versions
            continue
        remapped[key.replace("image_encoder.resnet.0.", "image_encoder.backbone.")] = value
    return remapped


def load_checkpoint(path, device):
    """Loads either a new-style checkpoint ({'model', 'model_cfg'}) or a legacy raw state_dict.

    The legacy format is the EfficientNet-B0 + MiniLM-L12 model from the original script;
    its embed_dim is read from the projection weights.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=True)

    if "model" in ckpt:
        cfg = ModelConfig(**ckpt["model_cfg"])
        state_dict = ckpt["model"]
    else:
        state_dict = _remap_legacy_keys(ckpt)
        cfg = ModelConfig(embed_dim=state_dict["image_encoder.fc.weight"].shape[0])

    model = CrossModalModel(cfg, pretrained=False)
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval()
