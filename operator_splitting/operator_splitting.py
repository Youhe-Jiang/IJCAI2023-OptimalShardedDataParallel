import time
import torch
from torch import nn
from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP

# Example:
# class Layer(nn.Module):
# def __init__(self, config):
# self.mlp = splitted_linear(config...)
# def forward(self, input):
# output = splitted_linear_forward(input, self.mlp, num_splits)

def operator_splitting(in_features, out_features, part_num, total_num, fsdp_level):
    # split operator into multiple parts
    new_in_features = int(in_features / total_num)
    if part_num < fsdp_level:
        partial_operator = FSDP(nn.Linear(new_in_features, out_features, bias=True))
    else:
        partial_operator = nn.Linear(new_in_features, out_features, bias=True)
    for param in partial_operator.parameters():
        param.requires_grad = True
    return partial_operator

def splitted_linear(in_features, out_features, total_num, fsdp_level):
    # define modulelist of splitted operators
    Modulelist = nn.ModuleList()
    for i in range(total_num):
        partial_operator = operator_splitting(in_features, out_features, i, total_num, fsdp_level)
        Modulelist.append(partial_operator)
    return Modulelist

def input_slice(input, part_num, total_num):
    # split the input data sequentially
    partial_features = int(input.size()[-1] / total_num)
    partial_input = input[:, :, (part_num*partial_features):(part_num*partial_features+partial_features)]
    return partial_input

def splitted_linear_forward(input, module, total_num):
    # define forward function of splitted operators
    output = 0
    for i in range(total_num):
        partial_input = input_slice(input, i, total_num)
        output += module[i](partial_input)
    return output
