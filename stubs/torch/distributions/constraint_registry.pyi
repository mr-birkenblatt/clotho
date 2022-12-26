from _typeshed import Incomplete


class ConstraintRegistry:
    def __init__(self) -> None: ...
    def register(self, constraint, factory: Incomplete | None = ...): ...
    def __call__(self, constraint): ...

biject_to: Incomplete
transform_to: Incomplete
