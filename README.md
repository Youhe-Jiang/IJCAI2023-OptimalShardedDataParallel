# OptimalShardedDataParallel

#### Description

Example of Optimal Sharded Data Parallel (OSDP) training GPT-2.

#### Environment
Cancel changes
- torch version 1.11.0+cu102
- fairscale version 0.4.5
- Memory limit: 8 GB

#### Model config

- 48 layers
- hidden size 1536

#### Implementation

Execute the  **train.py**  file through the  **script_gpt2_training.sh**  script, and deploy the OSDP experiment by specifying fsdp_type as OSDP (specify fsdp_type as FSDP to deploy the comparative experiment).

#### Experimental results

In OSDP, we maximize system throughput by maximizing device memory utilization. Since system throughput gain varies with the environment, and device memory utilization depends on the specific model, we demonstrate the device memory utilization of OSDP and FSDP tasks respectively.

- OSDP device memory utilization: 8057.35 MB / 8192 MB
- FSDP device memory utilization: 5656.91 MB / 8192 MB

