from typing import Iterable

from system.executors.executor import K, Service, V


class ProcessService(Service[K, V]):
    def __init__(self) -> None:
        pass

    def start(self) -> None:
        raise NotImplementedError()

    def is_alive(self) -> bool:
        raise NotImplementedError()

    def communicate(self, data: K) -> list[V]:
        raise NotImplementedError()

    @staticmethod
    def discover_services() -> Iterable['Service[K, V]']:
        raise NotImplementedError()
