MASTER_ADDR="162.105.146.119"
# NODE_RANK="0" # for 119
NODE_RANK="1" # for 118

torchrun --nnodes=2 --nproc_per_node=2 --master_addr $MASTER_ADDR --master_port 9991 --node_rank=$NODE_RANK train_gsdp_2nodes.py \
--train_batch_size 1 \
--fsdp_type AUTO_FSDP \
--vocab_size 30522 \
--hidden_size 2048 \
--num_hidden_layers 24 \
--num_attention_heads 16 \
--seq_length 512 \
--epochs 10 \
--lr 1e-4 \
--adam_weight_decay 0.01 \
--dropout_prob 0.1 \
--check_loss 0 \
--profile 0 \
--print_model 0
