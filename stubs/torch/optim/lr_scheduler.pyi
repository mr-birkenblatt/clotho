# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from _typeshed import Incomplete
from torch._six import inf as inf

from .optimizer import Optimizer as Optimizer


EPOCH_DEPRECATION_WARNING: str


class _LRScheduler:
    optimizer: Incomplete
    base_lrs: Incomplete
    last_epoch: Incomplete
    verbose: Incomplete

    def __init__(
        self, optimizer, last_epoch: int = ..., verbose: bool = ...): ...

    def state_dict(self): ...
    def load_state_dict(self, state_dict) -> None: ...
    def get_last_lr(self): ...
    def get_lr(self) -> None: ...

    def print_lr(
        self, is_verbose, group, lr,
        epoch: Incomplete | None = ...) -> None: ...

    o: Incomplete
    def step(self, epoch: Incomplete | None = ...): ...


class LambdaLR(_LRScheduler):
    optimizer: Incomplete
    lr_lambdas: Incomplete

    def __init__(
        self, optimizer, lr_lambda, last_epoch: int = ...,
        verbose: bool = ...) -> None: ...

    def state_dict(self): ...
    def load_state_dict(self, state_dict) -> None: ...
    def get_lr(self): ...


class MultiplicativeLR(_LRScheduler):
    optimizer: Incomplete
    lr_lambdas: Incomplete

    def __init__(
        self, optimizer, lr_lambda, last_epoch: int = ...,
        verbose: bool = ...) -> None: ...

    def state_dict(self): ...
    def load_state_dict(self, state_dict) -> None: ...
    def get_lr(self): ...


class StepLR(_LRScheduler):
    step_size: Incomplete
    gamma: Incomplete

    def __init__(
        self, optimizer, step_size, gamma: float = ...,
        last_epoch: int = ..., verbose: bool = ...) -> None: ...

    def get_lr(self): ...


class MultiStepLR(_LRScheduler):
    milestones: Incomplete
    gamma: Incomplete

    def __init__(
        self, optimizer, milestones, gamma: float = ...,
        last_epoch: int = ..., verbose: bool = ...) -> None: ...

    def get_lr(self): ...


class ConstantLR(_LRScheduler):
    factor: Incomplete
    total_iters: Incomplete

    def __init__(
        self, optimizer, factor=..., total_iters: int = ...,
        last_epoch: int = ..., verbose: bool = ...) -> None: ...

    def get_lr(self): ...


class LinearLR(_LRScheduler):
    start_factor: Incomplete
    end_factor: Incomplete
    total_iters: Incomplete

    def __init__(
        self, optimizer, start_factor=..., end_factor: float = ...,
        total_iters: int = ..., last_epoch: int = ...,
        verbose: bool = ...) -> None: ...

    def get_lr(self): ...


class ExponentialLR(_LRScheduler):
    gamma: Incomplete

    def __init__(
        self, optimizer, gamma, last_epoch: int = ...,
        verbose: bool = ...) -> None: ...

    def get_lr(self): ...


class SequentialLR(_LRScheduler):
    last_epoch: Incomplete
    optimizer: Incomplete

    def __init__(
        self, optimizer, schedulers, milestones, last_epoch: int = ...,
        verbose: bool = ...) -> None: ...

    def step(self) -> None: ...
    def state_dict(self): ...
    def load_state_dict(self, state_dict) -> None: ...


class CosineAnnealingLR(_LRScheduler):
    T_max: Incomplete
    eta_min: Incomplete

    def __init__(
        self, optimizer, T_max, eta_min: int = ..., last_epoch: int = ...,
        verbose: bool = ...) -> None: ...

    def get_lr(self): ...


class ChainedScheduler(_LRScheduler):
    optimizer: Incomplete
    def __init__(self, schedulers) -> None: ...
    def step(self) -> None: ...
    def state_dict(self): ...
    def load_state_dict(self, state_dict) -> None: ...


class ReduceLROnPlateau:
    factor: Incomplete
    optimizer: Incomplete
    min_lrs: Incomplete
    patience: Incomplete
    verbose: Incomplete
    cooldown: Incomplete
    cooldown_counter: int
    mode: Incomplete
    threshold: Incomplete
    threshold_mode: Incomplete
    best: Incomplete
    num_bad_epochs: Incomplete
    mode_worse: Incomplete
    eps: Incomplete
    last_epoch: int

    def __init__(
        self, optimizer, mode: str = ..., factor: float = ...,
        patience: int = ..., threshold: float = ...,
        threshold_mode: str = ..., cooldown: int = ..., min_lr: int = ...,
        eps: float = ..., verbose: bool = ...) -> None: ...

    def step(self, metrics, epoch: Incomplete | None = ...) -> None: ...
    @property
    def in_cooldown(self): ...
    def is_better(self, a, best): ...
    def state_dict(self): ...
    def load_state_dict(self, state_dict) -> None: ...


class CyclicLR(_LRScheduler):
    optimizer: Incomplete
    max_lrs: Incomplete
    total_size: Incomplete
    step_ratio: Incomplete
    mode: Incomplete
    gamma: Incomplete
    scale_fn: Incomplete
    scale_mode: str
    cycle_momentum: Incomplete
    base_momentums: Incomplete
    max_momentums: Incomplete
    base_lrs: Incomplete

    def __init__(
        self, optimizer, base_lr, max_lr, step_size_up: int = ...,
        step_size_down: Incomplete | None = ..., mode: str = ...,
        gamma: float = ..., scale_fn: Incomplete | None = ...,
        scale_mode: str = ..., cycle_momentum: bool = ...,
        base_momentum: float = ..., max_momentum: float = ...,
        last_epoch: int = ..., verbose: bool = ...) -> None: ...

    def get_lr(self): ...


class CosineAnnealingWarmRestarts(_LRScheduler):
    T_0: Incomplete
    T_i: Incomplete
    T_mult: Incomplete
    eta_min: Incomplete
    T_cur: Incomplete

    def __init__(
        self, optimizer, T_0, T_mult: int = ..., eta_min: int = ...,
        last_epoch: int = ..., verbose: bool = ...) -> None: ...

    def get_lr(self): ...
    last_epoch: Incomplete
    o: Incomplete
    def step(self, epoch: Incomplete | None = ...): ...


class OneCycleLR(_LRScheduler):
    optimizer: Incomplete
    total_steps: Incomplete
    anneal_func: Incomplete
    cycle_momentum: Incomplete
    use_beta1: Incomplete

    def __init__(
        self, optimizer, max_lr, total_steps: Incomplete | None = ...,
        epochs: Incomplete | None = ...,
        steps_per_epoch: Incomplete | None = ..., pct_start: float = ...,
        anneal_strategy: str = ..., cycle_momentum: bool = ...,
        base_momentum: float = ..., max_momentum: float = ...,
        div_factor: float = ..., final_div_factor: float = ...,
        three_phase: bool = ..., last_epoch: int = ...,
        verbose: bool = ...) -> None: ...

    def get_lr(self): ...
