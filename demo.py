import torch
import torch.nn as nn
import torch.nn.functional as F
from vi3nr import vi3nr_input_gain, vi3nr_gain_MC, vi3nr_uniform_

torch.manual_seed(0)

# Try different values for these
batch_size = 64
input_dim = 32
hidden_dim = 256
num_layers = 8
sigma_p = 1.0 # preactivation std for user to choose
act_nn = nn.Tanh # nn.Module version of the activation function
act_func = F.tanh # function version of the activation function
# act_nn = nn.Sigmoid
# act_func = F.sigmoid
input_data = torch.randn(batch_size, input_dim) * 20 + 5 # try different input data distributions

# Make MLP model
layers = [nn.Linear(input_dim, hidden_dim)]
for _ in range(num_layers-1):
    layers.append(act_nn())
    layers.append(nn.Linear(hidden_dim, hidden_dim))
model = nn.Sequential(*layers)

# Initialize model
# 1. Gain for first layer, uses statistics of input_data
first_layer_gain = vi3nr_input_gain(sigma_p=sigma_p, mean = input_data.mean(), std=input_data.std())
# 2. Gain for other layers, only uses sigma_p (statistics of previous preactivations)
gain = vi3nr_gain_MC(sigma_p=sigma_p, act_func = act_func)
for i in range(num_layers):
    if i == 0:
        vi3nr_uniform_(model[2*i].weight, gain=first_layer_gain)
    else:
        vi3nr_uniform_(model[2*i].weight, gain=gain)

# Print out preactivation std
preactivation_stds = []
with torch.no_grad():
    for i in range(num_layers):
        res = model[0:(2*i + 1)](input_data)
        preactivation_stds.append(res.std())
print("Preactivation standard deviation at each layer")
print(preactivation_stds)

