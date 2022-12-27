# pylint: disable=multiple-statements,unused-argument, invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias, unused-import
# pylint: disable=redefined-builtin,super-init-not-called, arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors, import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member, keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name, c-extension-no-member
# pylint: disable=protected-access,no-name-in-module, undefined-variable


from typing import Dict, Optional, Tuple

import torch
from _typeshed import Incomplete
from torch import Tensor


class LOBPCGAutogradFunction(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx, A: Tensor, k: Optional[int] = ..., B: Optional[Tensor] = ...,
        X: Optional[Tensor] = ..., n: Optional[int] = ...,
        iK: Optional[Tensor] = ..., niter: Optional[int] = ...,
        tol: Optional[float] = ..., largest: Optional[bool] = ...,
        method: Optional[str] = ..., tracker: None = ...,
        ortho_iparams: Optional[Dict[str, int]] = ...,
        ortho_fparams: Optional[Dict[str, float]] = ...,
        ortho_bparams: Optional[Dict[str, bool]] = ...) -> Tuple[
            Tensor, Tensor]: ...

    @staticmethod
    def backward(ctx, D_grad, U_grad): ...


def lobpcg(
    A: Tensor, k: Optional[int] = ..., B: Optional[Tensor] = ...,
    X: Optional[Tensor] = ..., n: Optional[int] = ...,
    iK: Optional[Tensor] = ..., niter: Optional[int] = ...,
    tol: Optional[float] = ..., largest: Optional[bool] = ...,
    method: Optional[str] = ..., tracker: None = ...,
    ortho_iparams: Optional[Dict[str, int]] = ...,
    ortho_fparams: Optional[Dict[str, float]] = ...,
    ortho_bparams: Optional[Dict[str, bool]] = ...) -> Tuple[
        Tensor, Tensor]: ...


class LOBPCG:
    A: Incomplete
    B: Incomplete
    iK: Incomplete
    iparams: Incomplete
    fparams: Incomplete
    bparams: Incomplete
    method: Incomplete
    tracker: Incomplete
    X: Incomplete
    E: Incomplete
    R: Incomplete
    S: Incomplete
    tvars: Incomplete
    ivars: Incomplete
    fvars: Incomplete
    bvars: Incomplete

    def __init__(
        self, A: Optional[Tensor], B: Optional[Tensor], X: Tensor,
        iK: Optional[Tensor], iparams: Dict[str, int], fparams: Dict[str,
                float], bparams: Dict[str, bool], method: str,
        tracker: None) -> None: ...

    def update(self) -> None: ...
    def update_residual(self) -> None: ...
    def update_converged_count(self): ...
    def stop_iteration(self): ...
    def run(self) -> None: ...
    def call_tracker(self) -> None: ...
