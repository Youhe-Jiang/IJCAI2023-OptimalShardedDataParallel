python -m torch.distributed.launch --nproc_per_node=8 --master_port 9999 train.py \
--train_batch_size 1 \
--vocab_size 30522 \
--hidden_size 1536 \
--num_hidden_layer 48 \
--num_attention_heads 48 \
--fsdp_type OSDP \
--device_memory_limit 8192 \
--profile 1 \
--print_model 1 \
