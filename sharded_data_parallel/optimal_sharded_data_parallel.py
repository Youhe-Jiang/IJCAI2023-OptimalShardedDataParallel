import torch
from sharded_data_parallel.osdp_modules.scheduler import Scheduler
from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP

def OptimalShardedDataParallel(model, model_description, device_information): 
    optimal_execution_plan = Scheduler(model_description, device_information)
    rank = torch.distributed.get_rank()
    if rank == 0:
        print('------------------------------------------')
        print('Model rebuilding...')
        print('------------------------------------------')
    model = model(model_description[2], optimal_execution_plan)
    return FSDP(model.cuda())
    
