import threading
import time
from typing import Generic, Iterable, TypeVar


K = TypeVar('K')
V = TypeVar('V')


class Service(Generic[K, V]):
    def start(self) -> None:
        raise NotImplementedError()

    def is_alive(self) -> bool:
        raise NotImplementedError()

    def communicate(self, data: K) -> list[V]:
        raise NotImplementedError()

    @staticmethod
    def discover_services() -> Iterable['Service[K, V]']:
        raise NotImplementedError()


class ExecutorManager(Generic[K, V]):
    def __init__(
            self,
            service_cls: Service[K, V],
            heartbeat: float,
            thread_count: int) -> None:
        self._service_cls = service_cls
        self._heartbeat = heartbeat
        self._thread_count = thread_count
        self._th: threading.Thread | None = None
        self._lock = threading.RLock()
        self._ensure_heartbeat()

    def _ensure_heartbeat(self) -> None:
        th = self._th
        if th is not None:
            return
        with self._lock:
            th = self._th
            if th is not None:
                return

            def heartbeat() -> None:
                try:
                    while True:
                        if cur_th is not self._th:
                            break
                        self.check_services()
                        time.sleep(self._heartbeat)
                finally:
                    if cur_th is self._th:
                        self._th = None

            cur_th = threading.Thread(target=heartbeat)
            self._th = cur_th
            cur_th.start()

    def check_services(self) -> None:
        for service in self._service_cls.discover_services():
            if service.is_alive():
                continue
            service.start()

    def communicate(self, data: K) -> list[V]:
        self._ensure_heartbeat()

        def get_result(tix: int, services: list[Service[K, V]]) -> None:
            for service in services:
                results[tix].extend(service.communicate(data))

        tcount = self._thread_count
        results: list[list[V]] = [[] for _ in range(tcount)]
        placements: list[list[Service[K, V]]] = [[] for _ in range(tcount)]
        for ix, service in enumerate(self._service_cls.discover_services()):
            placements[ix % tcount].append(service)
        ths = [
            threading.Thread(target=get_result, args=(tix, services,))
            for tix, services in enumerate(placements)
        ]
        for th in ths:
            th.start()
        for th in ths:
            th.join()
        return [
            res
            for arr in results
            for res in arr
        ]
