# OptimalShardedDataParallel

## Description

Optimal Sharded Data Parallel (OSDP), an automated parallel training system that combines the advantages from both data and model parallelism, which has a number of advanced characteristics:

- Efficiency: OSDP breaks the memory redundancy minimization limitation in previous ZeRO-based systems and enables to determine whether to perform parameter sharding for each operator individually. In addition, OSDP supports operator splitting and fine-grained memory management, enlarging the entire decision space.
- Flexibility: OSDP provides an efficient search engine to automatically find the optimal parallel strategies for each operator in the computation graph and eventually generates the execution plans.
- Usability:  only a few lines of code is needed to be replaced to complete the OSDP deployment.

Feel free to contribute codes, create issues and pull requests.

## Environment

The following command create the conda environment to be used:

```
$ conda env create -f environment.yml
```

Or prepare the environment by:

```
$ sh prepare_env.sh
```

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
  optim.step()
```

## Running OSDP

Execute the  **train_gpt2_...py**  file through the  **scripts/script_gpt2_...sh**  script, and deploy the OSDP experiment by specifying fsdp_type as OSDP (specify fsdp_type as FSDP to deploy the comparative experiment).

```
$ cd gpt
$ sh scripts/script_gpt2_osdp.sh
$ sh scripts/script_gpt2_fsdp.sh
```

## Experimental results

We show the system throughput and memory utilization of GPT-2 model training (48 layers with hidden_size 2048) in our environment (GPU memory limit: 8G):

- OSDP: 
  - device memory utilization: 8057.35 MB / 8192 MB  
  - overall system throughput: 192.2572615622423 seq/sec
- FSDP:
  - device memory utilization: 5656.91 MB / 8192 MB  
  - overall system throughput: 158.0486313692509 seq/sec

# Operator splitting

## Description

Operator splitting provides OSDP with the ability to search for a finer-grained execution plan for the model as well as minimizes memory surge in training, which provides OSDP with the ability to undertake a larger batch size and further optimize the system throughput.

## Implementation

Example using operator splitting:

```
class Layer(nn.Module):
  def __init__(self, config):
    self.mlp = splitted_linear(config...)
    ...
  
  def forward(self, input):
    output = splitted_linear_forward(input, self.mlp, num_splits)
    ...
```

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

Execute the  **train_bert_large...py**  file through the  **scripts/script_bert_large_...sh**  script, and deploy the Group Sharding or Communication with groups experiment by specifying fsdp_args as 'group_sharding' or 'communication_with_groups' (specify fsdp_args as 'none' to deploy the comparative experiment).

```
$ cd bert
$ sh scripts/script_bert_large_group_sharding.sh
$ sh scripts/script_bert_large_communication_with_groups.sh
$ sh scripts/script_bert_large_fsdp.sh
```

## Paper

Youhe Jiang, Xupeng Miao, Xiaonan Nie, Bin Cui. [OSDP: Optimal Sharded Data Parallel for Distributed Deep Learning](https://arxiv.org/abs/2209.13258). [ICML Hardware Aware Efficient Training (HAET) Workshop 2022](https://icml.cc/Conferences/2022/ScheduleMultitrack?event=13462#wse-detail-20407).

# Cite

If you use OSDP in a scientific publication, we would appreciate citations to the following paper:

```
@misc{https://doi.org/10.48550/arxiv.2209.13258,
  doi = {10.48550/ARXIV.2209.13258},
  
  url = {https://arxiv.org/abs/2209.13258},
  
  author = {Jiang, Youhe and Miao, Xupeng and Nie, Xiaonan and Cui, Bin},
  
  keywords = {Distributed, Parallel, and Cluster Computing (cs.DC), FOS: Computer and information sciences, FOS: Computer and information sciences},
  
  title = {OSDP: Optimal Sharded Data Parallel for Distributed Deep Learning},
  
  publisher = {arXiv},
  
  year = {2022},
  
  copyright = {Creative Commons Attribution 4.0 International}
}
```
