import torch
import torch.nn as nn
from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP

class Linear(nn.Module):
    def __init__(self, in_features, out_features, slice_size, fsdp_size):
        super(Linear, self).__init__()
        self.out_features = out_features
        self.slice_size = slice_size
        self.in_features_after_slice = in_features // slice_size
        self.module = nn.ModuleList()
        for i in range(slice_size):
            if i < fsdp_size:
                self.module.append(FSDP(nn.Linear(self.in_features_after_slice, self.out_features, bias=False)))
            else:
                self.module.append(nn.Linear(self.in_features_after_slice, self.out_features, bias=False))

    def forward(self, input):
        output = 0
        for i in range(self.slice_size):
            input_slice = input[:, self.in_features_after_slice*(i):self.in_features_after_slice*(i+1)]
            output += self.module[i](input_slice)
        return output
