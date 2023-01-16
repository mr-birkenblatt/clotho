from system.namespace.namespace import Namespace


BASE_CMD = "python -m system.namespace"


class ModuleBase:
    @staticmethod
    def module_name() -> str:
        raise NotImplementedError()

    def is_module_init(self) -> bool:
        raise NotImplementedError()

    def ensure_module_init(self, ns_name: str) -> None:
        if self.is_module_init():
            return
        raise ValueError(
            "module is not initialized!\n"
            f"run: {BASE_CMD} init "
            f"--namespace {ns_name} "
            f"--module {self.module_name()}")

    def from_namespace(
            self, other_namespace: Namespace, *, progress_bar: bool) -> None:
        raise ValueError(f"{self.module_name()} cannot be transferred")
