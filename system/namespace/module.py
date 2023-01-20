from misc.util import parent_python_module, python_module
from system.namespace.namespace import ModuleName, Namespace


class UnsupportedTransfer(ValueError):
    pass


class UnsupportedInit(ValueError):
    pass


class ModuleNotInitialized(ValueError):
    pass


MODULE_MAX_LEN = 20


class ModuleBase:
    @staticmethod
    def module_name() -> ModuleName:
        raise NotImplementedError()

    def initialize_module(self) -> None:
        raise UnsupportedInit(
            f"{self.module_name()} does not support initialization!")

    def is_module_init(self) -> bool:
        return True

    def ensure_module_init(self, ns_name: str) -> None:
        if self.is_module_init():
            return
        raise ModuleNotInitialized(
            "module is not initialized!\n"
            f"run: {parent_python_module(python_module())} init "
            f"--namespace {ns_name} "
            f"--module {self.module_name()}")

    def from_namespace(
            self, other_namespace: Namespace, *, progress_bar: bool) -> None:
        raise UnsupportedTransfer(
            f"{self.module_name()} cannot be transferred")


def get_module_obj(namespace: Namespace, module: ModuleName) -> ModuleBase:
    from system.embedding.store import EmbeddingStore, get_embed_store
    from system.links.store import get_link_store, LinkStore
    from system.msgs.store import get_message_store, MessageStore
    from system.users.store import get_user_store, UserStore

    if module == EmbeddingStore.module_name():
        return get_embed_store(namespace)
    if module == LinkStore.module_name():
        return get_link_store(namespace)
    if module == MessageStore.module_name():
        return get_message_store(namespace)
    if module == UserStore.module_name():
        return get_user_store(namespace)
    raise ValueError(f"unknown module: {module}")
