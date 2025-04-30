# VI3NR: Variance Informed Initialization for Implicit Neural Representations (_CVPR 2025_)

[Chamin Hewa Koneputugodage*](https://www.linkedin.com/in/chamin-hewa-koneputugodage-b3ba17148/), [Yizhak Ben-Shabat (Itzik)*](http://www.itzikbs.com), [Sameera Ramasinghe](https://www.linkedin.com/in/sameeraramasinghe/) and [Stephen Gould](http://users.cecs.anu.edu.au/~sgould/) from [ANU](https://www.anu.edu.au/), [Technion](https://www.technion.ac.il/) and [Pluralis AI](https://pluralis.ai/).

__[Project page](https://chumbyte.github.io/vi3nr-site/)&nbsp;/ [Arxiv](https://arxiv.org/abs/2504.19270)__

## Introduction
This is the code for Variance Informed Initialization for Implicit Neural Representations (VI3NR).

## Our initialization

Our initialization replaces
```py
from torch.nn.init import calculate_gain, xavier_uniform_
w = torch.empty(3, 5)
gain = calculate_gain('relu')
xavier_uniform_(w, gain=gain)
```
with (assuming `vi3nr.py` is in `sys.path`)

```py
from vi3nr import vi3nr_gain_MC, vi3nr_uniform_
w = torch.empty(3, 5)
sigma_p = 1
gain = vi3nr_gain_MC(sigma_p=sigma_p, act_func = F.relu)
vi3nr_uniform_(w, gain = gain)
```

We can calculate gain for any activation function, while pytorch has gain hardcoded for a few activation functions.
Note that
* calculating gain requires access to the activation function and the wanted standard deviation `sigma_p` (this assumes the previous layer is already initialized)
* we can also calculate gain given arbitrary input distribution for the layer (which we use for the first layer), see below

We explain how to properly use this in the next section.

## How to use

A standard approach to initializing networks is to ensure the variance of the preactivations at each layer is the same, and the variance of the gradients of loss w.r.t. the preactivations at each layer is the same.

**Forward Pass Initialization**
Our initialization gives a way to to set the preactivation at all layers to be a wanted value `sigma_p`.

For the first preactivation `z = x @ W0` where `x` is the input to the network and `W0` is the weights in the first layer, we initialize `W0` by
```
gain = vi3nr_input_gain(sigma_p=sigma_p, mean = x.mean(), std=x.std())
vi3nr_uniform_(W0, gain = gain)
```
where we have used the mean and std of `x` (should use the dataset mean and std).

For the preactivation at every other layer `z = act_func(z_prev) @ W`, where `z_prev` is the preactivation at the previous layer (which should have been set to `sigma_p`) and `W` is the weight at the current layer, we initialize `W` by

```
gain = vi3nr_gain_MC(sigma_p=sigma_p, act_func = act_func)
vi3nr_uniform_(W, gain = gain)
```
Look at `demo.py` to see this work for a basic MLP.

To see how this fares against pytorch's initializations, see `toy_experiments_vs_pytorch.py`. This follows the toy experiments in the paper 
(ability to maintaining variance for 100 layers and 1k units).

To run the toy experiments in the paper (more activations and initializations), see `toy_experiments_all.py`.

**Backward Pass Initialization**
We derive a condition to make the variance of the gradients of loss w.r.t. the preactivations at each layer is the same. Unlike previous initializations, this can be satisfied at the same time as the forward pass iniitalization, and we show it just becomes a condition on what `sigma_p` to choose. However it is non-trivial to directly solve for the optimal `sigma_p`, so we grid search for it instead.

You can see how to grid search in `toy_experiments_vs_pytorch.py` and `toy_experiments_all.py`.

**Choosing optimal `sigma_p`**
In practice, i.e. task performance, we find that the choice of `sigma_p` matters, and the optimal `sigma_p` is often close to the theoretical optimal for making the backward variance equal(see the paper). In particular, as the network gets deeper, the optimal `sigma_p` approaches the theoretical optimal (which makes sense, stable variance in the backward pass is more important for deeper networks). For shallower networks, we recommend grid searching around the theoretical optimal value (note that only a few iterations need to be run to determine whether a `sigma_p` is good or not).


## Thanks

Y. Ben-Shabat is supported by the Marie Sklodowska-Curie grant agreement No. 893465 S. Gould is a recipient of an ARC Future Fellowship (proj. no. LP200100421)
funded by the Australian Government.

## Citation

If you find our work useful in your research, please cite our paper:

```bibtex
@InProceedings{Koneputugodage2025vi3nr,
    author    = {Koneputugodage, Chamin Hewa and Ben-Shabat, Yizhak and Ramasinghe, Sameera and Gould, Stephen},
    title     = {VI3NR: Variance Informed Initialization for Implicit Neural Representations},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2025}
}
```