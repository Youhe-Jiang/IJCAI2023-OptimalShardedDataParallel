import torch
import functools
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from transformers import BertConfig, BertForPreTraining
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
from sharded_data_parallel.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
from sharded_data_parallel.gen_fsdp_args import gen_fsdp_args

def set_seed():
    seed = 123
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def model_forward(config, model, input_ids, attention_mask, token_type_ids, masked_lm_labels, next_sentence_label):
    outputs = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
    prediction_scores, seq_relationship_score = outputs.prediction_logits, outputs.seq_relationship_logits
    loss_fct = nn.CrossEntropyLoss(ignore_index = -1)
    masked_lm_loss = loss_fct(prediction_scores.view(-1, config.vocab_size), masked_lm_labels.view(-1))
    next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
    loss = masked_lm_loss + next_sentence_loss
    return loss, masked_lm_loss, next_sentence_loss

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
    config = BertConfig(vocab_size=args.vocab_size, 
                        hidden_size=args.hidden_size,
                        num_hidden_layers=args.num_hidden_layers, 
                        num_attention_heads=args.num_attention_heads, 
                        intermediate_size=args.hidden_size*4, 
                        max_position_embeddings=args.seq_length, 
                        attention_probs_dropout_prob=args.dropout_prob,
                        hidden_dropout_prob=args.dropout_prob)
    model = BertForPreTraining(config)
    model.to(device)
    fsdp_args = gen_fsdp_args(nnodes=2, nproc_per_node=2, gsdp_type='communication_with_groups', model_type='bert-large')
    with enable_wrap(wrapper_cls=FSDP, **fsdp_args):
        my_auto_wrap_policy = functools.partial(default_auto_wrap_policy, min_num_params=1e6)
        model = auto_wrap(model, auto_wrap_policy=my_auto_wrap_policy) #auto_wrap
    if args.profile and local_rank == 0:
        print_peak_memory("After creating model", local_rank, args.profile_type)
    model = FSDP(model, **fsdp_args) #wrap
    if args.profile and local_rank == 0:
        print_peak_memory("After creating FSDP model", local_rank, args.profile_type)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.adam_weight_decay)

    print("[Rank %d] Start training..."%rank)
    for ep in range(args.epochs):
        if not args.check_loss and not args.profile:
            trainloader = tqdm(trainloader)
        for iter, batch in enumerate(trainloader):
            start_time = time.time()
            input_ids, attention_mask, token_type_ids, masked_lm_labels, next_sentence_label= [tensor.to(device) for tensor in batch]
            if args.profile and local_rank == 0:
                torch.cuda.reset_peak_memory_stats(local_rank)
                print_peak_memory("\nBefore Forward", local_rank, args.profile_type)

            loss, masked_lm_loss, next_sentence_loss = \
                model_forward(config, model, input_ids, attention_mask, token_type_ids, masked_lm_labels, next_sentence_label)

            if args.profile and local_rank == 0:
                print_peak_memory("After Forward", local_rank, args.profile_type)

            loss.backward()

            if args.profile and local_rank == 0:
                print_peak_memory("After Backward", local_rank, args.profile_type)

            optimizer.step()

            if args.profile and local_rank == 0:
                print_peak_memory("After optimizer_step", local_rank, args.profile_type)

            optimizer.zero_grad()

            end_time = time.time()
            if args.check_loss or args.profile:
                print('[Rank %d | Epoch %d] (Iteration %d): Loss = %.3f, MLM_loss = %.3f, NSP_loss = %.6f, Time = %.3f'% \
                    (rank,ep,iter,loss.item(), masked_lm_loss.item(), next_sentence_loss.item(), end_time-start_time))

            if args.profile and iter >= 2:
                exit(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_batch_size", type=int, default=32, help="Training batch size for single GPU"
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
        "-s", "--seq_length", type=int, default=128, help="Maximum sequence len"
    )
    parser.add_argument(
        "--vocab_size", type=int, default=30522, help="Total number of vocab"
    )
    parser.add_argument(
        "--dropout_prob", type=float, default=0.1, help="Dropout rate."
    )
    parser.add_argument("--max_predictions_per_seq", type=int, default=20)
    parser.add_argument("-e", "--epochs", type=int,
                        default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate of adam")
    parser.add_argument(
        "--adam_weight_decay", type=float, default=0.01, help="Weight_decay of adam"
    )

    parser.add_argument(
        "--check_loss", type=int, default=0, help="Whether to check model correctness."
    )
    parser.add_argument(
        "--profile", type=int, default=0, help="Whether to profile model GPU memory."
    )
    parser.add_argument(
        "--profile_type", type=str, default='allocated', help="Profile allocated memory or reserved memory.",
        choices = ['allocated', 'reserved'],
    )

    args = parser.parse_args()
    set_seed()
    train(args)
