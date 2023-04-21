import enum
from typing import cast, get_args, Literal, TypedDict

import torch

from system.msgs.message import Message
from system.namespace.namespace import Namespace


ProviderRole = Literal["parent", "child"]
PROVIDER_PARENT: ProviderRole = "parent"
PROVIDER_CHILD: ProviderRole = "child"
PROVIDER_ROLES: list[ProviderRole] = list(get_args(ProviderRole))


class ProviderEnum(enum.Enum):
    PARENT = PROVIDER_PARENT
    CHILD = PROVIDER_CHILD


StorageMethod = Literal["array", "zip"]
VALID_STORAGE_METHODS: set[StorageMethod] = set(get_args(StorageMethod))

STORAGE_ARRAY: StorageMethod = "array"
STORAGE_COMPRESSED: StorageMethod = "zip"

STORAGE_ARRAY_ID = 0
STORAGE_COMPRESSED_ID = 1

STORAGE_MAP: dict[StorageMethod, int] = {
    STORAGE_ARRAY: STORAGE_ARRAY_ID,
    STORAGE_COMPRESSED: STORAGE_COMPRESSED_ID,
}


def parse_storage_method(text: str) -> StorageMethod:
    if text not in VALID_STORAGE_METHODS:
        raise ValueError(f"invalid storage method: {text}")
    return cast(StorageMethod, text)


def get_provider_role(role: str) -> ProviderRole:
    if role not in PROVIDER_ROLES:
        raise ValueError(f"{role} not a valid provider role")
    return cast(ProviderRole, role)


class EmbeddingProvider:
    def __init__(
            self,
            method: str,
            role: ProviderRole,
            embedding_name: str,
            embedding_hash: str,
            embedding_version: int,
            storage_method: StorageMethod) -> None:
        self._redis_name = f"{method}:{role}:{embedding_hash}"
        self._file_name = f"{method}.{role}"
        self._role = role
        self._embedding_name = embedding_name
        self._embedding_hash = embedding_hash
        self._embedding_version = embedding_version
        self._storage_method = storage_method

    def get_enum(self) -> ProviderEnum:
        if self._role == PROVIDER_CHILD:
            return ProviderEnum.CHILD
        if self._role == PROVIDER_PARENT:
            return ProviderEnum.PARENT
        raise ValueError(f"unknown role: {self._role}")

    def get_role(self) -> ProviderRole:
        return self._role

    def get_redis_name(self) -> str:
        return self._redis_name

    def get_file_name(self) -> str:
        return self._file_name

    def get_embedding_name(self) -> str:
        return self._embedding_name

    def get_embedding_hash(self) -> str:
        return self._embedding_hash

    def get_embedding_version(self) -> int:
        return self._embedding_version

    def get_storage_method(self) -> StorageMethod:
        return self._storage_method

    def get_embedding(self, msg: Message) -> torch.Tensor:
        raise NotImplementedError()

    @staticmethod
    def num_dimensions() -> int:
        raise NotImplementedError()


class NoEmbeddingProvider(EmbeddingProvider):
    def get_embedding(self, msg: Message) -> torch.Tensor:
        return torch.Tensor([0])

    @staticmethod
    def num_dimensions() -> int:
        return 1


EmbeddingProviderMap = TypedDict('EmbeddingProviderMap', {
    "parent": EmbeddingProvider,
    "child": EmbeddingProvider,
})


PROVIDER_CACHE: dict[Namespace, EmbeddingProviderMap] = {}


def get_embed_providers(namespace: Namespace) -> EmbeddingProviderMap:
    res = PROVIDER_CACHE.get(namespace)
    if res is None:
        res = create_embed_providers(namespace)
        PROVIDER_CACHE[namespace] = res
    return res


DBTransformerEmbeddingModule = TypedDict('DBTransformerEmbeddingModule', {
    "name": Literal["dbtransformer"],
    "conn": str,
    "model_hash": str,
    "storage": StorageMethod,
})
TransformerEmbeddingModule = TypedDict('TransformerEmbeddingModule', {
    "name": Literal["transformer"],
    "fname": str,
    "version": int,
    "is_harness": bool,
})
NoEmbeddingModule = TypedDict('NoEmbeddingModule', {
    "name": Literal["none"],
})
EmbeddingProviderModule = (
    DBTransformerEmbeddingModule |
    TransformerEmbeddingModule |
    NoEmbeddingModule)


def create_embed_providers(namespace: Namespace) -> EmbeddingProviderMap:
    pobj = namespace.get_embedding_providers()
    if pobj["name"] == "dbtransformer":
        from model.transformer_embed import load_db_providers

        db = namespace.get_db_connector(pobj["conn"])
        return load_db_providers(db, pobj["model_hash"], pobj["storage"])
    if pobj["name"] == "transformer":
        from model.transformer_embed import load_providers

        return load_providers(
            namespace.get_module_root("model"),
            pobj["fname"],
            pobj["version"],
            pobj["is_harness"])
    if pobj["name"] == "none":
        return {
            "parent": NoEmbeddingProvider(
                "none", "parent", "none", "none", 0, STORAGE_ARRAY),
            "child": NoEmbeddingProvider(
                "none", "child", "none", "none", 0, STORAGE_ARRAY),
        }
    raise ValueError(f"unknown embed provider: {pobj}")
