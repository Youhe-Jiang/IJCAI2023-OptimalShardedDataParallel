import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup
import typing
from typing import (
    TYPE_CHECKING,
    Any, 
    Callable,
    Dict,
    Generator,
    Iterator,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Set, 
    Tuple,
    Union,
    cast,
)

def dist_group_all_gather(
        output: Optional[torch.FloatTensor] = None, 
        inter_cache: Optional[List] = None,
        inter_cache_type: Optional[List] = None,
        input: Optional[torch.FloatTensor] = None,
        intra_group: Optional[ProcessGroup] = None, 
        inter_group: Optional[ProcessGroup] = None,
    ):
    # Define type of inter_cache
    for i in range(len(inter_cache_type)):
        if len(input) == inter_cache_type[i]:
            inter_cache = inter_cache[i]
    # Phase 1: Inter node communication
    dist._all_gather_base(inter_cache, input, group=inter_group)
    # Phase 2: Intra node communication
    dist._all_gather_base(output, inter_cache, group=intra_group)
