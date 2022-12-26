from _typeshed import Incomplete
from torch.nn.utils._expanded_weights.expanded_weights_impl import (
    ExpandedWeight as ExpandedWeight,
)
from torch.nn.utils._stateless import functional_call as functional_call


def call_for_per_sample_grads(module, batch_size, args, kwargs: Incomplete | None = ...): ...
