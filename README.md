# OptimalShardedDataParallel

#### Description

Example of Optimal Sharded Data Parallel (OSDP) training GPT-2.

#### Environment

- PyTorch version 1.11.0+cu102
- Fairscale version 0.4.5
- Memory limit: 8 GB

#### Model config

- 48 layers
- hidden size 1536
- 1.542B params

#### Implementation

- To implement OSDP deployment, we only need to replace a few lines of FSDP code: 
<img width="858" alt="image" src="https://user-images.githubusercontent.com/85312798/169662228-b6afe5ec-5d56-4aa7-92dd-6e6c12af456f.png">

- Execute the  **train.py**  file through the  **scripts/script_gpt2_training.sh**  script, and deploy the OSDP experiment by specifying fsdp_type as OSDP (specify fsdp_type as FSDP to deploy the comparative experiment).

#### Experimental results

In OSDP, we maximize overall system throughput by maximizing device memory utilization. Since system throughput varies with the environment, and device memory utilization depends on the specific model, we demonstrate the device memory utilization and overall system throughput (in our environment) of OSDP and FSDP tasks respectively.

- OSDP: 
  - device memory utilization: 8057.35 MB / 8192 MB  
  - overall system throughput: 192.2572615622423 seq/sec
- FSDP:
  - device memory utilization: 5656.91 MB / 8192 MB  
  - overall system throughput: 158.0486313692509 seq/sec
