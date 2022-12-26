import multiprocessing.pool

from .queue import SimpleQueue as SimpleQueue


def clean_worker(*args, **kwargs) -> None: ...

class Pool(multiprocessing.pool.Pool): ...
