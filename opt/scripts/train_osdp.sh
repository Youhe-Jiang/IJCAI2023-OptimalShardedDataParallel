torchrun --nproc_per_node=8 --master_port 9999 train_opt_osdp.py \
--train_batch_size 2 \
--epochs 10 \
--lr 1e-4 \
--dropout_prob 0.1 \
--profile 1
