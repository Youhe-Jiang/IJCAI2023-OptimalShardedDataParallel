# OptimalShardedDataParallel

#### Description

Example of Optimal Sharded Data Parallel (OSDP) training GPT-2.

#### Environment

- torch version 1.11.0+cu102
- fairscale version 0.4.5
- Memory limit: 8 GB

#### Model config

- 48 layers
- hidden size 1536

#### Implementation

Execute the  **train.py**  file through the  **script_gpt2_training.sh**  script, and deploy the OSDP experiment by specifying fsdp_type as OSDP (specify fsdp_type as FSDP to deploy the comparative experiment).

#### Experimental results

- OSDP memory utilization: 8057.35 MB / 8192 MB
- FSDP memory utilization: 5656.91 MB / 8192 MB
- OSDP system throughput gain: 23.4%

