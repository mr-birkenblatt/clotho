# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from typing import Callable, Optional, Tuple, Union

import torch.testing
from _typeshed import Incomplete
from torch.types import _TensorOrTensors


class GradcheckError(RuntimeError):
    ...


def get_numerical_jacobian(
    fn, inputs, target: Incomplete | None = ..., eps: float = ...,
    grad_out: float = ...): ...


def get_numerical_jacobian_wrt_specific_input(
    fn, input_idx, inputs, outputs, eps, input: Incomplete | None = ...,
    is_forward_ad: bool = ...) -> Tuple[torch.Tensor, ...]: ...


def get_analytical_jacobian(
    inputs, output, nondet_tol: float = ..., grad_out: float = ...): ...


def gradcheck(
    func: Callable[..., Union[_TensorOrTensors]], inputs: _TensorOrTensors,
    *, eps: float = ..., atol: float = ..., rtol: float = ...,
    raise_exception: bool = ..., check_sparse_nnz: bool = ...,
    nondet_tol: float = ..., check_undefined_grad: bool = ...,
    check_grad_dtypes: bool = ..., check_batched_grad: bool = ...,
    check_batched_forward_grad: bool = ..., check_forward_ad: bool = ...,
    check_backward_ad: bool = ..., fast_mode: bool = ...) -> bool: ...


def gradgradcheck(
    func: Callable[..., _TensorOrTensors], inputs: _TensorOrTensors,
    grad_outputs: Optional[_TensorOrTensors] = ..., *, eps: float = ...,
    atol: float = ..., rtol: float = ...,
    gen_non_contig_grad_outputs: bool = ..., raise_exception: bool = ...,
    nondet_tol: float = ..., check_undefined_grad: bool = ...,
    check_grad_dtypes: bool = ..., check_batched_grad: bool = ...,
    check_fwd_over_rev: bool = ..., check_rev_over_rev: bool = ...,
    fast_mode: bool = ...) -> bool: ...
