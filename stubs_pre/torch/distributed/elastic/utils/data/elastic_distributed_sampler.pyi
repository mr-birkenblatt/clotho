# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access,no-name-in-module,undefined-variable


from _typeshed import Incomplete
from torch.utils.data.distributed import (
    DistributedSampler as DistributedSampler,
)


class ElasticDistributedSampler(DistributedSampler):
    start_index: Incomplete
    num_samples: Incomplete
    total_size: Incomplete

    def __init__(
        self, dataset, num_replicas: Incomplete | None = ...,
        rank: Incomplete | None = ..., start_index: int = ...) -> None: ...

    def __iter__(self): ...
    def __len__(self): ...
