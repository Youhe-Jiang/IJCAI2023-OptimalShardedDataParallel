from Scheduler import Scheduler
from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP

def OSDP(model, model_description, device_information): 
    optimal_execution_plan = Scheduler(model_description, device_information)
    model = model(model_description[2], optimal_execution_plan)
    return FSDP(model.cuda())
    
