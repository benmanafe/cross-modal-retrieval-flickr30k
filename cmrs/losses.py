import torch
import torch.nn.functional as F


def clip_loss(image_embeds, text_embeds, logit_scale):
    """Symmetric InfoNCE: matching pairs sit on the diagonal of the batch similarity matrix.

    Computed in float32 even under autocast, since exp(logit_scale) can reach 100.
    """
    image_embeds = image_embeds.float()
    text_embeds = text_embeds.float()

    logits = image_embeds @ text_embeds.t() * logit_scale.exp()
    labels = torch.arange(logits.size(0), device=logits.device)

    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.t(), labels)
    return (loss_i2t + loss_t2i) / 2
