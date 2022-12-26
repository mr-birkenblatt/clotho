from enum import Enum

from _typeshed import Incomplete
from torch import Tensor
from torch._prims.utils import DimsSequenceType, TensorLikeType


class RETURN_TYPE(Enum):
    NEW: Incomplete
    VIEW: Incomplete
    INPLACE: Incomplete

class ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND(Enum):
    DEFAULT: Incomplete
    ALWAYS_BOOL: Incomplete
    COMPLEX_TO_FLOAT: Incomplete

abs: Incomplete
acos: Incomplete
acosh: Incomplete
asin: Incomplete
atan: Incomplete
cos: Incomplete
cosh: Incomplete
bessel_i0e: Incomplete
bessel_i1e: Incomplete
bitwise_not: Incomplete
cbrt: Incomplete
ceil: Incomplete
digamma: Incomplete
erf: Incomplete
erf_inv: Incomplete
erfc: Incomplete
exp: Incomplete
expm1: Incomplete
floor: Incomplete
is_finite: Incomplete
is_infinite: Incomplete
lgamma: Incomplete
log: Incomplete
log1p: Incomplete
log2: Incomplete
reciprocal: Incomplete
neg: Incomplete
round: Incomplete
sign: Incomplete
sin: Incomplete
sinh: Incomplete
sqrt: Incomplete
square: Incomplete
tan: Incomplete
tanh: Incomplete
add: Incomplete
atan2: Incomplete
bitwise_and: Incomplete
bitwise_or: Incomplete
bitwise_xor: Incomplete
div: Incomplete
eq: Incomplete
ge: Incomplete
gt: Incomplete
igamma: Incomplete
igammac: Incomplete
le: Incomplete
lt: Incomplete
maximum: Incomplete
minimum: Incomplete
mul: Incomplete
ne: Incomplete
nextafter: Incomplete
pow: Incomplete
rsqrt: Incomplete
shift_left: Incomplete
shift_right_arithmetic: Incomplete
shift_right_logical: Incomplete
as_strided: Incomplete
broadcast_in_dim: Incomplete
collapse_view: Incomplete

def expand_dims(a: TensorLikeType, dimensions: DimsSequenceType) -> TensorLikeType: ...

slice: Incomplete
slice_in_dim: Incomplete
split_dim: Incomplete
squeeze: Incomplete
transpose: Incomplete
view_of: Incomplete

def collapse(a: Tensor, start: int, end: int) -> Tensor: ...

concatenate: Incomplete
reshape: Incomplete
rev: Incomplete
select: Incomplete
clone: Incomplete
convert_element_type: Incomplete
device_put: Incomplete
to_dtype: Incomplete
copy_to: Incomplete
resize: Incomplete
sum: Incomplete
prod: Incomplete
amax: Incomplete
amin: Incomplete
all: Incomplete
any: Incomplete
empty: Incomplete
empty_like: Incomplete
full: Incomplete
full_like: Incomplete
