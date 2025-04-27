# mypy: allow-untyped-defs
"""
Variance-informed initialization utilities for neural network parameters.

Extends PyTorch’s default initializers with custom gain calculations
based on desired preactivation variance, activation function and/or
layer input statistics.
"""
import math
from typing import Optional as _Optional

import torch
from torch import Tensor
from torch.nn.init import _no_grad_uniform_, _no_grad_normal_, _calculate_fan_in_and_fan_out

def vi3nr_input_gain(
        sigma_p: float,
        mean: float,
        std: float,
) -> float:
    """
    Compute the gain for the weights of the layer so that the layer has the 
    desired preactivation standard deviation (`sigma_p`). Uses statistics of
    the input distribution to the layer.

    Usually used for the first layer.

    Parameters
    ----------
    sigma_p : float
        Desired standard deviation of the preactivation of the layer.
    mean : float
        Mean of the input distribution to the layer.
    std : float
        Standard deviation of the input distribution to the layer.

    Returns
    -------
    float
        Gain factor used to scale the initial weights so that the resulting
        preactivation has standard deviation `sigma_p`.
    """
    return math.sqrt(sigma_p ** 2 / (mean**2 + std**2))

def vi3nr_gain_MC(
        sigma_p: float,
        act_func,
        samples = 1_000_000,
        device = None,
) -> float:
    """
    Compute the gain for the weights of the layer so that the layer has the 
    desired preactivation standard deviation (`sigma_p`). Assumes the 
    preactivations of the previous layer have mean 0 and std `sigma_p`.
    Uses Monte Carlo estimation to estimate postactivation statistics.

    Usually used for layers other than the first layer.

    Parameters
    ----------
    sigma_p : float
        Desired standard deviation of the preactivation of the layer.
    act_func : Callable
        Activation function (e.g., torch.nn.functional.relu) applied to inputs.
    samples : int, optional
        Number of Monte Carlo samples to draw (default: 1_000_000).
    device : torch.device or None, optional
        Device on which to do the Monte Carlo estimation (default: None).

    Returns
    -------
    float
        Gain factor used to scale the initial weights so that the resulting
        preactivation has standard deviation `sigma_p`.
    """
    mc_samples = torch.randn(samples, device=device) * sigma_p
    fx = act_func(mc_samples)
    mu = fx.mean()
    sigma2 = fx.var()
    return (sigma_p ** 2 / (mu**2 + sigma2)).sqrt().item()


def vi3nr_uniform_(
    tensor: Tensor,
    gain: float = 1.0,
    generator: _Optional[torch.Generator] = None,
) -> Tensor:
    """
    In-place initialization of `tensor` with values drawn from a uniform distribution
    scaled by a custom gain, following the VI3NR scheme.

    This is analogous to PyTorch's `xavier_uniform_` but uses a variance-informed gain.

    Parameters
    ----------
    tensor : torch.Tensor
        The tensor to be initialized in-place.
    gain : float, optional
        Scaling factor for initialization (default: 1.0).
    generator : torch.Generator or None, optional
        Random number generator for reproducibility (default: None).

    Returns
    -------
    torch.Tensor
        The initialized tensor with values in `[-a, a]`, where
        `a = gain * sqrt(3 / fan_in)`.
    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain / math.sqrt(float(fan_in))
    a = math.sqrt(3.0) * std

    return _no_grad_uniform_(tensor, -a, a, generator)


def vi3nr_normal_(
    tensor: Tensor,
    gain: float = 1.0,
    generator: _Optional[torch.Generator] = None,
) -> Tensor:
    """
    In-place initialization of `tensor` with values drawn from a normal distribution
    scaled by a custom gain, following the VI3NR scheme.

    This is analogous to PyTorch's `xavier_normal_` but uses a variance-informed gain.

    Parameters
    ----------
    tensor : torch.Tensor
        The tensor to be initialized in-place.
    gain : float, optional
        Scaling factor for initialization (default: 1.0).
    generator : torch.Generator or None, optional
        Random number generator for reproducibility (default: None).

    Returns
    -------
    torch.Tensor
        The initialized tensor with mean 0 and standard deviation
        `gain / sqrt(fan_in)`.
    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain / math.sqrt(float(fan_in))

    return _no_grad_normal_(tensor, 0.0, std, generator)



