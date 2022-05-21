import torch
from torch import nn
import numpy as np
from data_parallel.OSDP_modules.Search_Engine import SearchAlg
from data_parallel.OSDP_modules.Profiler_output import mem_map, time_map, extra_overhead

class Search(nn.Module):
    def __init__(self, max_mem, mem_cost_list, time_cost_list, bsz):
        self.max_mem = max_mem
        self.num_layers = len(mem_cost_list)
        self.num_strategies = len(mem_cost_list[0])
        self.mem_cost_list = mem_cost_list
        self.time_cost_list = time_cost_list
        self.bsz = bsz

    def run(self):
        dp = SearchAlg(self.max_mem, self.num_layers, self.num_strategies)
        dp.set_v_and_cost(self.mem_cost_list, self.time_cost_list)
        comm_cost, res_list, mem_remain = dp.fit()
        if res_list == None:
            return 0, 0
        else:
            return res_list, comm_cost


Cand_plan = []
Cand_cost = []
def Scheduler(model_description, device_information):
    rank = torch.distributed.get_rank()
    num_layers = model_description[0]
    hidden_size = [512, 1024, 1536, 2048, 3072, 4096, 6144, 8192]
    n_feat = hidden_size.index(model_description[1])
    device_memory_limit_without_extra_overhead = device_information - extra_overhead
    for bsz in range(1,4):
        if rank == 0:
            print('------------------------------------------')
            print('Current training batch size: ', bsz)
            print('Start searching for optimal execution plan...')
            print('------------------------------------------')
        time_cost_list = np.array([time_map[n_feat][bsz][:]]*num_layers)
        mem_cost_list = np.trunc(np.array([mem_map[n_feat][bsz][:]]*num_layers)).astype(np.int64)
        res_list, cost= Search(device_memory_limit_without_extra_overhead, mem_cost_list, time_cost_list, bsz).run()
        if res_list == 0:
            if rank == 0:
                print('========Minimum possible memory exceeds device memory limit========')
            break
        else:
            Cand_plan.append(res_list)
            Cand_cost.append(cost)
    if rank == 0:
        print('------------------------------------------')
        print('Complete optimal execution plan search.')
        print('------------------------------------------')
    return Cand_plan[Cand_cost.index(min(Cand_cost))]
