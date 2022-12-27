# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from _typeshed import Incomplete
from torch._prims import utils as utils


meta_lib: Incomplete


def check(b, s) -> None: ...


def toRealValueType(dtype): ...


def meta_index_select(self, dim, index): ...


def meta_index_select_out(self, dim, index, out): ...


def meta_abs(self): ...


def meta_abs_out(self, out): ...


def meta_max(self): ...


def meta_min(self): ...


def squareCheckInputs(self, f_name) -> None: ...


def checkUplo(uplo: str): ...


def meta_linalg_eigh(self, uplo: str = ...): ...


def meta_pad2d(self, padding): ...


def meta_dot(self, tensor): ...


def meta_var_mean_correction(
    self, dim, *, correction, keepdim: bool = ...): ...


def meta_inverse(self): ...


def meta_bernoulli(self, *, generator: Incomplete | None = ..., out): ...


def meta_adaptive_avg_pool2d(self, output_size): ...


def meta_adaptive_avg_pool3d(self, output_size): ...
