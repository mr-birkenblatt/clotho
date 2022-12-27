# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from _typeshed import Incomplete


class _ThreadingDataLoader2:
    threads: Incomplete
    datapipes: Incomplete
    collate_fn: Incomplete

    def __init__(
        self, datapipe, num_workers: int = ...,
        collate_fn: Incomplete | None = ...) -> None: ...

    def __iter__(self): ...
    def __del__(self) -> None: ...


class DataLoader2:

    def __new__(
        cls, dataset, batch_size: int = ...,
        shuffle: Incomplete | None = ..., sampler: Incomplete | None = ...,
        batch_sampler: Incomplete | None = ..., num_workers: int = ...,
        collate_fn: Incomplete | None = ..., pin_memory: bool = ...,
        drop_last: bool = ..., timeout: int = ...,
        worker_init_fn: Incomplete | None = ..., *,
        prefetch_factor: int = ..., persistent_workers: bool = ...,
        batch_outside_worker: bool = ..., parallelism_mode: str = ...): ...
