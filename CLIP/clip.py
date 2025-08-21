import torch
from torch import nn
import math
from pathlib import Path
from dataclasses import dataclass
from torchvision import transforms
from tokenizers import Tokenizer
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image
from transformers import RobertaTokenizer, RobertaModel, ViTImageProcessor, ViTForImageClassification, ViTFeatureExtractor
import pandas as pd


df = pd.read_csv('')

@dataclass
class args:
    img_size = (224, 224)
    bls = 77
    bs = 32
    emb_dims = 768
    proj_dims = 768
    attn_dr = 0.1
    nh = 12
    dr = 0.1
    ep = 100
    lr = 1e-4
    dl = 12
    wei_decay = 0.2
    b1 = 0.9
    b2 = 0.98
    eps = 1e-6
    device = 'cuda'
    vs = 2000
    head_lr = 1e-3
    img_enc_lr = 1e-4
    text_enc_lr = 1e-5


class norm(nn.Module):
    def __init__(self, emb_dims: int = args.emb_dims):
        super().__init__()
        self.ln = nn.LayerNorm(emb_dims)

    def forward(self, x):
        return self.ln(x)


tok = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaModel.from_pretrained('roberta-base')

# class text(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.tok = tok
#         self.model = model
#         self.ln = norm
#         self.proj = nn.Linear(args.emb_dims, args.proj_dims, device=args.device)
#
#         for x in self.model.parameters():
#             x.requires_grad = True
#         self.model.train()
#
#     def forward(self, x):
#         x['input_ids'] = x['input_ids'].squeeze(1)
#         x


