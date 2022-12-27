# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access,no-name-in-module,undefined-variable


import os
from typing import Any, Dict, Optional, Union

import torch.overrides
from _typeshed import Incomplete
from torch.package import Importer as Importer
from torch.package import PackageExporter as PackageExporter
from torch.package import PackageImporter as PackageImporter
from torch.package import sys_importer as sys_importer

from ._compatibility import compatibility as compatibility
from .graph import Graph as Graph
from .graph import PythonCode as PythonCode


class _EvalCacheLoader:
    eval_cache: Incomplete
    next_id: int
    def __init__(self) -> None: ...
    def cache(self, src: str, globals: Dict[str, Any]): ...
    def get_source(self, module_name) -> Optional[str]: ...


def reduce_graph_module(
    body: Dict[Any, Any], import_block: str) -> torch.nn.Module: ...


def reduce_package_graph_module(
    importer: PackageImporter, body: Dict[Any, Any],
    generated_module_name: str) -> torch.nn.Module: ...


def reduce_deploy_graph_module(
    importer: PackageImporter, body: Dict[
            Any, Any], import_block: str) -> torch.nn.Module: ...


class _WrappedCall:
    cls: Incomplete
    cls_call: Incomplete
    def __init__(self, cls, cls_call) -> None: ...
    def __call__(self, obj, *args, **kwargs): ...


class GraphModule(torch.nn.Module):
    def __new__(cls, *args, **kwargs): ...
    training: Incomplete

    def __init__(
        self, root: Union[torch.nn.Module, Dict[str, Any]], graph: Graph,
        class_name: str = ...): ...

    __jit_unused_properties__: Incomplete
    @property
    def graph(self) -> Graph: ...
    @graph.setter
    def graph(self, g: Graph) -> None: ...

    def to_folder(
        self, folder: Union[str, os.PathLike], module_name: str = ...): ...

    def add_submodule(self, target: str, m: torch.nn.Module) -> bool: ...
    def delete_submodule(self, target: str) -> bool: ...
    def delete_all_unused_submodules(self) -> None: ...
    @property
    def code(self) -> str: ...
    def recompile(self) -> PythonCode: ...
    def __reduce_deploy__(self, importer: Importer): ...
    def __reduce_package__(self, exporter: PackageExporter): ...
    def __reduce__(self): ...
    def __deepcopy__(self, memo): ...
    def __copy__(self): ...
