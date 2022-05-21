import sys 
sys.path.insert(0, '..')
import h5py
import time
import torch
import random
import argparse
import functools
import numpy as np
from torch import nn
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader
from memory_utils import print_peak_memory
from AdditionDataset import AdditionDataset
from torch.utils.data.distributed import DistributedSampler
from mingpt.osdp_model import GPTConfig, CausalSelfAttention
from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP

def set_seed():
    seed = 123
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def train(args):
    if args.fsdp_type == 'OSDP':
        from OptimalShardedDataParallel import OSDP
        from mingpt.osdp_model import GPT, GPTConfig, CausalSelfAttention
    elif args.fsdp_type == 'FSDP':
        from mingpt.fsdp_model import GPT, GPTConfig, CausalSelfAttention
    torch.distributed.init_process_group(backend="nccl")
    local_rank = args.local_rank
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    ndigit = 2
    train_dataset = AdditionDataset(ndigit=ndigit, split='train')
    test_dataset = AdditionDataset(ndigit=ndigit, split='test')
    mconf = GPTConfig(
            train_dataset.vocab_size, 
            train_dataset.block_size, 
            n_layer=args.num_hidden_layers, 
            n_head=args.num_attention_heads, 
            n_embd=args.hidden_size,
    )
    # ======== Construct OSDP model ========
    if args.fsdp_type == 'OSDP':
        model_description = [args.num_hidden_layers, args.hidden_size, mconf]
        device_information = args.device_memory_limit
        model = GPT
        model = OSDP(model, model_description, device_information)
    # ======== Construct FSDP model ========
    elif args.fsdp_type == 'FSDP':
        model = GPT(mconf)
        model = FSDP(model.cuda())
    my_lr = 6e-4
    if rank == 0 and args.print_model:
        print(model)
    optim = Adam(model.parameters(), lr=my_lr)
    data = train_dataset
    loader = DataLoader(
            data, 
            shuffle=True, 
            pin_memory=True, 
            batch_size=args.train_batch_size*args.seq_length, 
            num_workers=world_size,
    )
    pbar = tqdm(enumerate(loader), total=len(loader))
    for it, (x, y) in pbar:
        if it == 4:
            start_time = time.time()
        if it == 7:
            end_time = time.time()
        x = x.cuda()
        y = y.cuda()
        out, loss = model(x, y)
        loss.backward()
        optim.step()
        if it >= 7:
            if rank == 0 and args.profile == 1:
                print('System throughput: ', 
                        args.train_batch_size * args.seq_length / ((end_time-start_time) / 3))
                print_peak_memory("Peak memory", rank, args.profile_type)
            exit(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--local_rank" ,type=int,default=-1)
    parser.add_argument(
        "--train_batch_size", type=int, default=32, help="Training batch size for single GPU"
    )
    parser.add_argument(
        "--device_memory_limit", type=int, default=8192, help="Device memory limit"
    )
    parser.add_argument(
        "--fsdp_type", type=str, default='None', help="Type of FSDP"
    )
    parser.add_argument(
        "--hidden_size", type=int, default=768, help="Hidden size of transformer model",
    )
    parser.add_argument(
        "--num_hidden_layers", type=int, default=12, help="Number of layers"
    )
    parser.add_argument(
        "-a",
        "--num_attention_heads",
        type=int,
        default=12,
        help="Number of attention heads",
    )
    parser.add_argument(
        "-s", "--seq_length", type=int, default=96, help="Maximum sequence len"
    )
    parser.add_argument(
        "--vocab_size", type=int, default=30522, help="Total number of vocab"
    )
    parser.add_argument(
        "--profile", type=int, default=0, help="Whether to profile model GPU memory"
    )
    parser.add_argument(
        "--profile_type", type=str, default='allocated', help="Profile allocated memory or reserved memory",
        choices = ['allocated', 'reserved'],
    )
    parser.add_argument(
        "--print_model", type=int, default=0, help="Whether to print model"
    )
    args = parser.parse_args()
    set_seed()
    train(args)
