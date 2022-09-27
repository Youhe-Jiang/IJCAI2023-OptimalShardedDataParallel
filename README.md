# OptimalShardedDataParallel

## Description

Optimal Sharded Data Parallel (OSDP), an automated parallel training system that combines the advantages from both data and model parallelism.

## Environment

- torch version 1.11.0+cu102
- fairscale version 0.4.5
- device memory limit: 8 GB

## Implementation

Example using OSDP:

```
from data_parallel.optimal_sharded_data_parallel import OptimalShardedDataParallel as OSDP
...
sharded_module = OSDP(my_module, model_description, device_information)
optim = torch.optim.Adam(sharded_module.parameters(), lr=0.0001)
for sample, label in dataload.next_batch:
  out = sharded_module(x=sample, y=3, z=torch.Tensor([1]))
  loss = criterion(out, label)
  loss.backward()
  optim.step(
```

## Running OSDP

Execute the  **train.py**  file through the  **scripts/script_gpt2_training.sh**  script, and deploy the OSDP experiment by specifying fsdp_type as OSDP (specify fsdp_type as FSDP to deploy the comparative experiment).

```
$ sh scripts/script_gpt2_training.sh
```

## Experimental results

### GPT configs:

- 48 layers
- hidden size 1536
- 1.542B params

In OSDP, we maximize overall system throughput by maximizing device memory utilization. Since system throughput varies with the environment, and device memory utilization depends on the specific model, we demonstrate the device memory utilization and overall system throughput (in our environment) of OSDP and FSDP tasks respectively.

- OSDP: 
  - device memory utilization: 8057.35 MB / 8192 MB  
  - overall system throughput: 192.2572615622423 seq/sec
- FSDP:
  - device memory utilization: 5656.91 MB / 8192 MB  
  - overall system throughput: 158.0486313692509 seq/sec

# New features: Group Sharding & Communication with groups

## Group Sharding

- Stage 1: Intra-group Sharded Data Parallel.
- Stage 2: Inter-group All-Reduce.

Trade-off between memory consumption and system throughput for more efficient use of inter-machine bandwidth.

## Communication with groups

- Stage 1: Intra-group All-Gather and Reduce-Scatter during the Sharded Data Parallel process.
- Stage 2: Inter-group All-Gather and Reduce-Scatter during the Sharded Data Parallel process.

Increase system throughput by reducing inter-machine communication parameters (usually the inter-machine bandwidth is much lower than the intra-machine bandwidth).

## Implementation

Example of using Group Sharding training bert-large with 2 machines and 4 GPUs (we use the auto_wrap API provided by fairscale to complete sharded data parallel deployment):

```
fsdp_args = gen_fsdp_args(nnodes=2, nproc_per_node=2, gsdp_type='group_sharding', model_type='bert-large')
with enable_wrap(wrapper_cls=FSDP, **fsdp_args):
        my_auto_wrap_policy = functools.partial(default_auto_wrap_policy, min_num_params=1e6)
        model = auto_wrap(model, auto_wrap_policy=my_auto_wrap_policy) #auto_wrap
```

Example of using Communication with groups training bert-large with 2 machines and 4 GPUs:

```
fsdp_args = gen_fsdp_args(nnodes=2, nproc_per_node=2, gsdp_type='communication_with_groups', model_type='bert-large')
with enable_wrap(wrapper_cls=FSDP, **fsdp_args):
        my_auto_wrap_policy = functools.partial(default_auto_wrap_policy, min_num_params=1e6)
        model = auto_wrap(model, auto_wrap_policy=my_auto_wrap_policy) #auto_wrap
```

## Running Group Sharding & Communication with groups

We provide an example of OSDP training bert with Group Sharding & Communication with groups:

Execute the  **train_bert.py**  file through the  **scripts/script_bert_training.sh**  script, and deploy the Group Sharding or Communication with groups experiment by specifying fsdp_args as 'group_sharding' or 'communication_with_groups' (specify fsdp_args as 'none' to deploy the comparative experiment).

```
$ sh scripts/script_bert_training.sh
```
