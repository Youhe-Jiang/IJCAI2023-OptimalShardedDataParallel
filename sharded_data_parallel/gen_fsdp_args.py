import torch
import torch.distributed as dist 

def gen_inter_cache(model_type: str, nnodes: int):
    inter_cache_params = []
    inter_cache_grads = []
    if model_type == 'bert-large':
        arr_bert_params = [165327, 7813632, 262400, 1049600, 1048832, 7821263]
        arr_bert_grads = [1574400, 1573632, 690127, 7821263, 7813632]
        for i in range(len(arr_bert_params)):
            inter_cache_params.append(torch.zeros(arr_bert_params[i]*nnodes, dtype=torch.float32).cuda())
        for i in range(len(arr_bert_grads)):
            inter_cache_grads.append(torch.zeros(arr_bert_grads[i]*nnodes, dtype=torch.float32).cuda())
        return arr_bert_params, arr_bert_grads, inter_cache_params, inter_cache_grads

def gen_fsdp_args(nnodes: int, nproc_per_node: int, gsdp_type: str, model_type: str):
    rank = dist.get_rank()
    group_intra = []
    group_inter = []
    group_arr_intra = []
    group_arr_inter = []
    for i in range(nnodes):
        group_arr_intra.append([])
    for i in range(nproc_per_node):
        group_arr_inter.append([])
    for i in range(nnodes):
        for j in range(nproc_per_node):
            group_arr_intra[i].append(i*nproc_per_node+j)
    for i in range(nproc_per_node):
        for j in range(nnodes):
            group_arr_inter[i].append(j*nproc_per_node+i)
    for i in range(nnodes):
        group_intra.append(dist.new_group(group_arr_intra[i]))
    for i in range(nproc_per_node):
        group_inter.append(dist.new_group(group_arr_inter[i]))
    process_group_intra_node = group_intra[rank//nproc_per_node]
    process_group_inter_node = group_inter[rank%nproc_per_node]
    if gsdp_type == 'group_sharding':
        print('/* GROUP SHARDING MODE WITH FSDP_DEG:', nproc_per_node, 'AND DP_DEG:', nnodes, '*/')
        return dict(
                process_group=process_group_intra_node,
                process_group_reduce_scatter=process_group_intra_node,
                process_group_all_reduce=process_group_inter_node, 
                all_reduce_before_backward=True,
               )
    elif gsdp_type == 'communication_with_groups':
        print('/* FSDP ALL_GATHER AND REDUCE_SCATTER IN GROUPS */')
        # TODO: Define inter_cache automatically.
        # Specific inter_cache for bert-large
        inter_cache_type, inter_cache_reduce_scatter_type, \
                inter_cache, inter_cache_reduce_scatter = gen_inter_cache(model_type, nnodes=2)
        return dict(
                process_group_intra_node=process_group_intra_node,
                process_group_inter_node=process_group_inter_node,
                inter_cache=inter_cache,
                inter_cache_type=inter_cache_type,
                inter_cache_reduce_scatter_type=inter_cache_reduce_scatter_type,
                inter_cache_reduce_scatter=inter_cache_reduce_scatter,
                communication_with_groups=True,
               )
    else:
        return dict()
