# OptimalShardedDataParallel

## Description

Example of Optimal Sharded Data Parallel (OSDP) training GPT-2.

## Environment

- torch version 1.11.0+cu102
- fairscale version 0.4.5
- device memory limit: 8 GB

## Model config

- 48 layers
- hidden size 1536
- 1.542B params

## Implementation

OSDP deployment:

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

In OSDP, we maximize overall system throughput by maximizing device memory utilization. Since system throughput varies with the environment, and device memory utilization depends on the specific model, we demonstrate the device memory utilization and overall system throughput (in our environment) of OSDP and FSDP tasks respectively.

- OSDP: 
  - device memory utilization: 8057.35 MB / 8192 MB  
  - overall system throughput: 192.2572615622423 seq/sec
- FSDP:
  - device memory utilization: 5656.91 MB / 8192 MB  
  - overall system throughput: 158.0486313692509 seq/sec
