import os, sys, time
import math
import random
import numpy as np
import torch

import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

def smape(pred, tar):
    if torch.is_tensor(tar):
        denom = pred.abs() + tar.abs()
    else:
        denom = pred.abs() + np.abs(tar)
    # return (2 * (pred-tar).abs() / denom).mean()
    smapes = ((pred-tar).abs() / denom).mean()
    smapes[torch.isnan(smapes)] = 1.0
    return smapes

def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For all GPUs
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_derivative(f, x):
    x.requires_grad_(True)
    y = f(x)
    y.backward(torch.ones_like(x))
    res = x.grad.clone()
    x.grad = None
    x.requires_grad_(False)
    return res

sine = lambda x: torch.sin(30*x)
gaussian = lambda x: torch.exp(-x**2/(2*0.05**2))
sinc = lambda x: torch.sin(x)/(x+1e-15)
wavelet = lambda x: torch.cos(x)*torch.exp(-x**2)

device = torch.device("cuda")

#####################################################################
### Using fixed values for common activation functions
#####################################################################

print("Using fixed values for common activation functions")

configs = [
    ('xavier/kumar-tanh', 1, F.tanh, 1),
    ('pytorch-tanh', 1, F.tanh, 2.78),
    ('ours1-tanh', 1, F.tanh, 2.54),
    ('oursB-tanh', 0.1, F.tanh, 1.02),
    ('kumar-sigmoid', 1, F.sigmoid, 12.96),
    ('pytorch-sigmoid', 1, F.sigmoid, 1),
    ('ours1-sigmoid', 1, F.sigmoid, 3.41),
    ('oursB-sigmoid', 6.8, F.sigmoid, 104.28),
    ('all-relu', 1, F.relu, 2),
    ]

print("Method | Forward Error (SMAPE) | Backward Error (SMAPE) |")
for config in configs:

    name, sigma_p, act, gain_sq = config
    forward_outs = []
    backward_outs = []
    bs = 10000 # batch size, reduce this if running out of GPU memory!
    hls = 1000 # hidden layer size
    set_all_seeds(1)
    pre_0 = torch.randn(bs, hls, device=device) * sigma_p
    pre_0.requires_grad=True
    pre = pre_0
    for i in range(100):
        post = act(pre)
        l = torch.nn.Linear(hls, hls, bias=False).to(device)
        init.xavier_uniform_(l.weight, gain=np.sqrt(gain_sq))
        pre = l(post)
    set_all_seeds(1)
    grads = torch.randn(bs, hls).to(device)
    loss = (pre*grads).sum()
    loss.backward()

    forward_smape = smape(pre.detach().var(dim=-1), sigma_p**2)
    backward_smape = smape(pre_0.grad.detach().var(dim=-1), 1)
    print(f"{name: <20} {forward_smape*100:.1f}\t {backward_smape*100:.1f}")


#####################################################################
### Using fixed values for common activation functions
#####################################################################

print("Gain^2 and Backward Condition Value for different activation functions, sigma_p=1")

print("Method | Forward Error (SMAPE) | Backward Error (SMAPE) | Gain^2 | Backwards Cond.")

sigma_p = 1

names = ['tanh', 'sigmoid', 'relu', 'sine', 'gaussian', 'sinc', 'wavelet']
act_list = [F.tanh, F.sigmoid, F.relu, sine, gaussian, sinc, wavelet]
for act_i, act in enumerate(act_list):
    act_prime = lambda x: compute_derivative(act, x)
    name = names[act_i]
    num_samples = 1000000
    set_all_seeds(1)
    mc_samples = torch.randn(num_samples, device=device) * sigma_p
    fx = act(mc_samples)
    mu2_f = fx.mean()**2
    sigma2_f = fx.var()
    fprimex = act_prime(mc_samples)
    mu2_fprime = fprimex.mean()**2
    sigma2_fprime = fprimex.var()
    cond = sigma_p**2 * (mu2_fprime + sigma2_fprime) / (mu2_f + sigma2_f)
    gain_sq = sigma_p**2 / (mu2_f + sigma2_f)



    forward_outs = []
    backward_outs = []
    bs = 10000 # batch size, reduce this if running out of GPU memory!
    hls = 1000 # hidden layer size
    set_all_seeds(1)
    pre_0 = torch.randn(bs, hls, device=device) * sigma_p
    pre_0.requires_grad=True
    pre = pre_0
    for i in range(100):
        post = act(pre)
        # post = F.sigmoid(pre)
        l = torch.nn.Linear(hls, hls, bias=False).to(device)
        init.xavier_uniform_(l.weight, gain=math.sqrt(gain_sq))
        pre = l(post)
    set_all_seeds(1)
    grads = torch.randn(bs, hls).to(device)
    loss = (pre*grads).sum()
    loss.backward()

    forward_smape = smape(pre.detach().var(dim=-1), sigma_p**2)
    backward_smape = smape(pre_0.grad.detach().var(dim=-1), 1)
    print(f"{name: <20} {forward_smape*100:.1f}\t {backward_smape*100:.1f}\t\t\t {gain_sq:.3f}\t\t {cond:.5f}")


#####################################################################
### Searching for sigma_p to satisfy backwards cond (cond ~= 1)
#####################################################################

print("Searching for sigma_p that best satisfies backward cond (cond ~= 1)")

values = torch.arange(0.001, 10+1e-5, 0.001)

print("Method | Optimal sigma_p | Gain^2 | Backwards Cond. | Forward Error (SMAPE) | Backward Error (SMAPE)")

names = ['tanh', 'sigmoid', 'relu', 'sine', 'gaussian', 'sinc', 'wavelet']
act_list = [F.tanh, F.sigmoid, F.relu, sine, gaussian, sinc, wavelet]
for act_i, act in enumerate(act_list):
    act_prime = lambda x: compute_derivative(act, x)
    name = names[act_i]

    t0 = time.time()
    conds = torch.zeros_like(values)
    gain_sqs = torch.zeros_like(values)
    num_samples = 1000000
    for i, sigma_p in enumerate(values):
        set_all_seeds(1)
        mc_samples = torch.randn(num_samples, device=device) * sigma_p
        fx = act(mc_samples)
        mu2_f = fx.mean()**2
        sigma2_f = fx.var()
        fprimex = act_prime(mc_samples)
        mu2_fprime = fprimex.mean()**2
        sigma2_fprime = fprimex.var()
        cond = sigma_p**2 * (mu2_fprime + sigma2_fprime) / (mu2_f + sigma2_f)
        gain_sq = sigma_p**2 / (mu2_f + sigma2_f)
        conds[i] = cond
        gain_sqs[i] = gain_sq

    if torch.isnan(conds).any():
        import pdb; pdb.set_trace()
    idx = (conds-1).abs().argmin()
    # idx = (values-1).abs().argmin()
    best_cond = conds[idx]
    best_sigma_p = values[idx]
    best_gain_sq = gain_sqs[idx]
    time_taken = time.time() - t0


    forward_outs = []
    backward_outs = []
    bs = 10000 # batch size, reduce this if running out of GPU memory!
    hls = 1000 # hidden layer size
    set_all_seeds(1)
    pre_0 = torch.randn(bs, hls, device=device) * best_sigma_p
    pre_0.requires_grad=True
    pre = pre_0
    for i in range(100):
        post = act(pre)
        l = torch.nn.Linear(hls, hls, bias=False).to(device)
        init.xavier_uniform_(l.weight, gain=math.sqrt(best_gain_sq))
        pre = l(post)
    set_all_seeds(1)
    grads = torch.randn(bs, hls).to(device)
    loss = (pre*grads).sum()
    loss.backward()

    forward_smape = smape(pre.detach().var(dim=-1), best_sigma_p**2)
    backward_smape = smape(pre_0.grad.detach().var(dim=-1), 1)
    print(f"{name: <12}   {best_sigma_p:.3f} \t   {best_gain_sq:.3f}\t{best_cond:.5f} \t {forward_smape*100:.1f} \t\t\t {backward_smape*100:.1f}"
    f"\t\t\t [took {time_taken:.3f}s for search over {len(values)} values]")