# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access,no-name-in-module,undefined-variable


from typing import Tuple, TypeVar

from torch.utils.data.datapipes.datapipe import MapDataPipe


T_co = TypeVar('T_co', covariant=True)


class ConcaterMapDataPipe(MapDataPipe):
    datapipes: Tuple[MapDataPipe]
    length: int
    def __init__(self, *datapipes: MapDataPipe) -> None: ...
    def __getitem__(self, index) -> T_co: ...
    def __len__(self) -> int: ...


class ZipperMapDataPipe(MapDataPipe[Tuple[T_co, ...]]):
    datapipes: Tuple[MapDataPipe[T_co], ...]
    length: int
    def __init__(self, *datapipes: MapDataPipe[T_co]) -> None: ...
    def __getitem__(self, index) -> Tuple[T_co, ...]: ...
    def __len__(self) -> int: ...
