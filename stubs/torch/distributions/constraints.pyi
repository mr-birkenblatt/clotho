from _typeshed import Incomplete


class Constraint:
    is_discrete: bool
    event_dim: int
    def check(self, value) -> None: ...

class _Dependent(Constraint):
    def __init__(self, *, is_discrete=..., event_dim=...) -> None: ...
    @property
    def is_discrete(self): ...
    @property
    def event_dim(self): ...
    def __call__(self, *, is_discrete=..., event_dim=...): ...
    def check(self, x) -> None: ...

def is_dependent(constraint): ...

class _DependentProperty(property, _Dependent):
    def __init__(self, fn: Incomplete | None = ..., *, is_discrete=..., event_dim=...) -> None: ...
    def __call__(self, fn): ...

class _IndependentConstraint(Constraint):
    base_constraint: Incomplete
    reinterpreted_batch_ndims: Incomplete
    def __init__(self, base_constraint, reinterpreted_batch_ndims) -> None: ...
    @property
    def is_discrete(self): ...
    @property
    def event_dim(self): ...
    def check(self, value): ...

class _Boolean(Constraint):
    is_discrete: bool
    def check(self, value): ...

class _OneHot(Constraint):
    is_discrete: bool
    event_dim: int
    def check(self, value): ...

class _IntegerInterval(Constraint):
    is_discrete: bool
    lower_bound: Incomplete
    upper_bound: Incomplete
    def __init__(self, lower_bound, upper_bound) -> None: ...
    def check(self, value): ...

class _IntegerLessThan(Constraint):
    is_discrete: bool
    upper_bound: Incomplete
    def __init__(self, upper_bound) -> None: ...
    def check(self, value): ...

class _IntegerGreaterThan(Constraint):
    is_discrete: bool
    lower_bound: Incomplete
    def __init__(self, lower_bound) -> None: ...
    def check(self, value): ...

class _Real(Constraint):
    def check(self, value): ...

class _GreaterThan(Constraint):
    lower_bound: Incomplete
    def __init__(self, lower_bound) -> None: ...
    def check(self, value): ...

class _GreaterThanEq(Constraint):
    lower_bound: Incomplete
    def __init__(self, lower_bound) -> None: ...
    def check(self, value): ...

class _LessThan(Constraint):
    upper_bound: Incomplete
    def __init__(self, upper_bound) -> None: ...
    def check(self, value): ...

class _Interval(Constraint):
    lower_bound: Incomplete
    upper_bound: Incomplete
    def __init__(self, lower_bound, upper_bound) -> None: ...
    def check(self, value): ...

class _HalfOpenInterval(Constraint):
    lower_bound: Incomplete
    upper_bound: Incomplete
    def __init__(self, lower_bound, upper_bound) -> None: ...
    def check(self, value): ...

class _Simplex(Constraint):
    event_dim: int
    def check(self, value): ...

class _Multinomial(Constraint):
    is_discrete: bool
    event_dim: int
    upper_bound: Incomplete
    def __init__(self, upper_bound) -> None: ...
    def check(self, x): ...

class _LowerTriangular(Constraint):
    event_dim: int
    def check(self, value): ...

class _LowerCholesky(Constraint):
    event_dim: int
    def check(self, value): ...

class _CorrCholesky(Constraint):
    event_dim: int
    def check(self, value): ...

class _Square(Constraint):
    event_dim: int
    def check(self, value): ...

class _Symmetric(_Square):
    def check(self, value): ...

class _PositiveSemidefinite(_Symmetric):
    def check(self, value): ...

class _PositiveDefinite(_Symmetric):
    def check(self, value): ...

class _Cat(Constraint):
    cseq: Incomplete
    lengths: Incomplete
    dim: Incomplete
    def __init__(self, cseq, dim: int = ..., lengths: Incomplete | None = ...) -> None: ...
    @property
    def is_discrete(self): ...
    @property
    def event_dim(self): ...
    def check(self, value): ...

class _Stack(Constraint):
    cseq: Incomplete
    dim: Incomplete
    def __init__(self, cseq, dim: int = ...) -> None: ...
    @property
    def is_discrete(self): ...
    @property
    def event_dim(self): ...
    def check(self, value): ...

dependent: Incomplete
dependent_property: Incomplete
independent: Incomplete
boolean: Incomplete
nonnegative_integer: Incomplete
positive_integer: Incomplete
integer_interval: Incomplete
real: Incomplete
real_vector: Incomplete
positive: Incomplete
greater_than: Incomplete
greater_than_eq: Incomplete
less_than: Incomplete
multinomial: Incomplete
unit_interval: Incomplete
interval: Incomplete
half_open_interval: Incomplete
simplex: Incomplete
lower_triangular: Incomplete
lower_cholesky: Incomplete
corr_cholesky: Incomplete
square: Incomplete
symmetric: Incomplete
positive_semidefinite: Incomplete
positive_definite: Incomplete
cat: Incomplete
stack: Incomplete
