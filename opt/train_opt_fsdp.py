import torch
import functools
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from transformers import OPTConfig, OPTForCausalLM 
from dataloader import DataLoaderForBert
import argparse
from tqdm import tqdm
import numpy as np
import random
import h5py
import time
from torch.utils.data.distributed import DistributedSampler
from fairscale.nn import auto_wrap, default_auto_wrap_policy, enable_wrap, wrap
import sys
sys.path.insert(0, '..')
from utils.memory_utils import print_peak_memory
from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP

def set_seed():
    seed = 123
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def train(args):
    torch.distributed.init_process_group(backend="nccl")
    rank = torch.distributed.get_rank()
    local_rank = rank
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    print('rank:',rank,'local_rank:', local_rank)

    print("[Rank %d] Creating Dataloader..."%rank)
    dataset = DataLoaderForBert()
    trainloader = DataLoader(dataset=dataset,
                            batch_size=args.train_batch_size,
                            pin_memory = True,
                            sampler=DistributedSampler(dataset,shuffle=False))

    print("[Rank %d] Creating Model..."%rank)

    # Initializing a OPT facebook/opt-2.7b style configuration
    configuration = OPTConfig()
    configuration.ffn_dim = 28672 
    configuration.hidden_size = 7168
    configuration.num_attention_heads = 56
    configuration.num_hidden_layers = 8
    
    # Initializing a model (with random weights) from the facebook/opt-large style configuration
    model = OPTForCausalLM(configuration)
    # # [Optional] Download model from facebook/opt- ...
    # model = OPTForCausalLM.from_pretrained("facebook/opt-125m")
    # model = OPTForCausalLM.from_pretrained("facebook/opt-350m")
    # model = OPTForCausalLM.from_pretrained("facebook/opt-1.3b")
    # model = OPTForCausalLM.from_pretrained("facebook/opt-2.7b")
    # model = OPTForCausalLM.from_pretrained("facebook/opt-6.7b")
    # model = OPTForCausalLM.from_pretrained("facebook/opt-13b")
    # model = OPTForCausalLM.from_pretrained("facebook/opt-30b")
    
    # # Layer by layer wrap 
    # for i in range(configuration.num_hidden_layers):
    #     model.model.decoder.layers[i] = FSDP(model.model.decoder.layers[i])
    
    # Op by op wrap
    with enable_wrap(wrapper_cls=FSDP):
        my_auto_wrap_policy = functools.partial(default_auto_wrap_policy, min_num_params=1e6)
        model = auto_wrap(model, auto_wrap_policy=my_auto_wrap_policy) #auto_wrap
    model = FSDP(model) # wrap
    
    # Load model to GPU
    model.to(device)

    # Check model configuration
    # if rank == 0:
    #     print(model)
    
    if args.profile and local_rank == 0:
        print_peak_memory("After creating FSDP model", local_rank, args.profile_type)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.adam_weight_decay)

    print("[Rank %d] Start training..."%rank)
    start_iter = 4
    end_iter = 7
    for ep in range(args.epochs):
        if not args.profile:
            trainloader = tqdm(trainloader)
        for iter, batch in enumerate(trainloader):
            if args.profile:
                if iter == start_iter:
                    total_start_time = time.time()
                elif iter == end_iter:
                    total_end_time = time.time()
                    avg_time = (total_end_time-total_start_time)/(end_iter-start_iter)
                    if rank == 0:
                        print("Average iteration time is: %.4f s"%avg_time)

            input_ids, attention_mask, token_type_ids, masked_lm_labels, next_sentence_label= [tensor.to(device) for tensor in batch]
            if args.profile and local_rank == 0 and iter <= 2:
                torch.cuda.reset_peak_memory_stats(local_rank)
                print_peak_memory("\nBefore Forward", local_rank, args.profile_type)

            loss = model(input_ids=input_ids, labels=token_type_ids).loss
            
            if args.profile and local_rank == 0 and iter <= 2:
                print_peak_memory("After Forward", local_rank, args.profile_type)

            loss.backward()

            if args.profile and local_rank == 0 and iter <= 2:
                print_peak_memory("After Backward", local_rank, args.profile_type)

            optimizer.step()

            if args.profile and local_rank == 0 and iter <= 2:
                print_peak_memory("After optimizer_step", local_rank, args.profile_type)

            if rank == 0:
                print('[Epoch %d] (Iteration %d): Loss = %.3f'% (ep,iter,loss))
            optimizer.zero_grad()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_batch_size", type=int, default=32, help="Training batch size for single GPU"
    )
    parser.add_argument(
        "--dropout_prob", type=float, default=0.1, help="Dropout rate."
    )
    parser.add_argument("-e", "--epochs", type=int,
                        default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate of adam")
    parser.add_argument(
        "--profile", type=int, default=0, help="Whether to profile model GPU memory."
    )
    parser.add_argument(
        "--profile_type", type=str, default='allocated', help="Profile allocated memory or reserved memory.",
        choices = ['allocated', 'reserved'],
    )
    parser.add_argument(
        "--adam_weight_decay", type=float, default=0.01, help="Weight_decay of adam."
    )
    args = parser.parse_args()
    set_seed()
    train(args)
