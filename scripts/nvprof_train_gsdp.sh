nvprof --profile-child-processes -o fsdp_test_short_%p.nvvp torchrun --nproc_per_node=4 --master_port 9999 train_gsdp.py \
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
