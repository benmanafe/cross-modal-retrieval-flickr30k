import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from transformers import AutoModel


class ImageEncoder(nn.Module):
    def __init__(self, embed_dim=256):
        super(ImageEncoder, self).__init__()
        resnet = models.efficientnet_b0(pretrained=True)
    
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)

        for param in self.resnet.parameters():
            param.requires_grad = False

        for param in self.resnet[0][6:].parameters():
            param.requires_grad = True

        self.dropout = nn.Dropout(p=0.5)
        
        self.fc = nn.Linear(1280, embed_dim)

    def forward(self, images):
        features = self.resnet(images) # (Batch, 512, 1, 1)
        features = features.view(features.size(0), -1) # flatten to (Batch, 512)
        features = self.dropout(features)
        embeddings = self.fc(features) # project to (Batch, embed_dim)
        embeddings = F.normalize(embeddings, dim=-1)
        return embeddings
    
    
class TextEncoder(nn.Module):
    def __init__(self, embed_dim=1024):
        super(TextEncoder, self).__init__()
        self.transformer = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L12-v2")

        for param in self.transformer.parameters():
            param.requires_grad = False

        for param in self.transformer.encoder.layer[8:].parameters():
            param.requires_grad = True

        hidden_size = self.transformer.config.hidden_size
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(hidden_size, embed_dim)

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # text_features = outputs.last_hidden_state[:, 0, :] # CLS token
        token_embeddings = outputs.last_hidden_state
        
        expanded_mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        masked_embeddings = token_embeddings * expanded_mask
        sum_embeddings = torch.sum(masked_embeddings, 1)
        sum_mask = torch.sum(expanded_mask, 1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_pooled_features = sum_embeddings / sum_mask
        mean_pooled_features = self.dropout(mean_pooled_features)
        
        # embeddings = self.fc(text_features)
        embeddings = self.fc(mean_pooled_features)
        embeddings = F.normalize(embeddings, dim=-1)
        return embeddings
    
    
class CrossModalModel(nn.Module):
    def __init__(self, embed_dim=1024):
        super(CrossModalModel, self).__init__()
        self.image_encoder = ImageEncoder(embed_dim)
        self.text_encoder = TextEncoder(embed_dim)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1/0.07))

    def forward(self, images, input_ids, attention_mask):
        image_embeds = self.image_encoder(images)
        text_embeds = self.text_encoder(input_ids, attention_mask)
        return image_embeds, text_embeds