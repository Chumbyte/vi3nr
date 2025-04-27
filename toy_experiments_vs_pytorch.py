import os, sys, time
import math
import random
import numpy as np
import torch

import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from vi3nr import vi3nr_input_gain, vi3nr_gain_MC, vi3nr_uniform_


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For all GPUs
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def smape(pred, tar):
    if torch.is_tensor(tar):
        denom = pred.abs() + tar.abs()
    else:
        denom = pred.abs() + np.abs(tar)
    # return (2 * (pred-tar).abs() / denom).mean()
    smapes = ((pred-tar).abs() / denom).mean()
    smapes[torch.isnan(smapes)] = 1.0
    return smapes

device = torch.device("cuda")

configs = [
    ('pytorch-tanh', 'tanh', F.tanh),
    ('ours-tanh', 'tanh', F.tanh),
    ('pytorch-sigmoid', 'sigmoid', F.sigmoid),
    ('ours-sigmoid', 'sigmoid', F.sigmoid),
    ('pytorch-relu', 'relu', F.relu),
    ('ours-relu', 'relu', F.relu),
    ]

print("Method | Forward Error (SMAPE) | Backward Error (SMAPE) |")
sigma_p = 1
for config in configs:
    name, act_name, act = config
    forward_outs = []
    backward_outs = []
    bs = 10000 # batch size, reduce this if running out of GPU memory!
    hls = 1000 # hidden layer size
    set_all_seeds(1)
    pre_0 = torch.randn(bs, hls, device=device) * sigma_p
    pre_0.requires_grad=True
    pre = pre_0
    
    if 'pytorch' in name:
        gain = init.calculate_gain(act_name)
    elif 'ours' in name:
        gain=vi3nr_gain_MC(sigma_p=1, act_func = act)
    else:
        raise
    
    for i in range(100):
        post = act(pre)
        l = torch.nn.Linear(hls, hls, bias=False).to(device)
        if 'pytorch' in name:
            init.xavier_uniform_(l.weight, gain=gain)
        elif 'ours' in name:
            vi3nr_uniform_(l.weight, gain=gain)
        else:
            raise
        pre = l(post)
    set_all_seeds(1)
    grads = torch.randn(bs, hls).to(device)
    loss = (pre*grads).sum()
    loss.backward()

    forward_smape = smape(pre.detach().var(dim=-1), sigma_p**2)
    backward_smape = smape(pre_0.grad.detach().var(dim=-1), 1)
    print(f"{name: <20} {forward_smape*100:.1f}\t {backward_smape*100:.1f}")



