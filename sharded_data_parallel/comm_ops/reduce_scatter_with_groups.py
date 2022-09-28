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

def dist_group_reduce_scatter(
        output: Optional[torch.FloatTensor] = None, 
        inter_cache: Optional[List] = None,
        inter_cache_type: Optional[List] = None,
        input: Optional[torch.FloatTensor] = None,
        intra_group: Optional[ProcessGroup] = None,
        inter_group: Optional[ProcessGroup] = None,
    ) -> None:
    # Define type of inter_cache
    for i in range(len(inter_cache_type)):
        if len(output) == inter_cache_type[i]:
            inter_cache = inter_cache[i]
    # Phase 1: Intra node communication 
    dist._reduce_scatter_base(inter_cache, input, group=intra_group)
    # Phase 2: Inter node communication
    dist._reduce_scatter_base(output, inter_cache, group=inter_group)
