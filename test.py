import torch
from model import NNSR


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

model = NNSR(35,160)
print(count_parameters(model))