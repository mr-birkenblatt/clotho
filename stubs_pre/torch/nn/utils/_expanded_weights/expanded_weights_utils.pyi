# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access,no-name-in-module,undefined-variable


import torch

from .expanded_weights_impl import ExpandedWeight as ExpandedWeight


def standard_kwargs(kwarg_names, expanded_args): ...


def forward_helper(func, expanded_args, expanded_kwargs): ...


def set_grad_sample_if_exists(
    maybe_expanded_weight, per_sample_grad_fn) -> None: ...


def unpack_expanded_weight_or_tensor(maybe_expanded_weight, func=...): ...


def sum_over_all_but_batch_and_last_n(
    tensor: torch.Tensor, n_dims: int) -> torch.Tensor: ...
