from _typeshed import Incomplete
from torch.utils.data.distributed import (
    DistributedSampler as DistributedSampler,
)


class ElasticDistributedSampler(DistributedSampler):
    start_index: Incomplete
    num_samples: Incomplete
    total_size: Incomplete
    def __init__(self, dataset, num_replicas: Incomplete | None = ..., rank: Incomplete | None = ..., start_index: int = ...) -> None: ...
    def __iter__(self): ...
    def __len__(self): ...
