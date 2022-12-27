# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from argparse import ArgumentParser

from _typeshed import Incomplete
from torch.distributed.argparse_util import check_env as check_env
from torch.distributed.argparse_util import env as env
from torch.distributed.elastic.multiprocessing import Std as Std
from torch.distributed.elastic.multiprocessing.errors import record as record
from torch.distributed.elastic.utils import macros as macros
from torch.distributed.elastic.utils.logging import get_logger as get_logger
from torch.distributed.launcher.api import LaunchConfig as LaunchConfig


        elastic_launch as elastic_launch
from typing import Callable, List, Tuple, Union


log: Incomplete


def get_args_parser() -> ArgumentParser: ...


def parse_args(args): ...


def parse_min_max_nnodes(nnodes: str): ...


def determine_local_world_size(nproc_per_node: str): ...


def get_rdzv_endpoint(args): ...


def get_use_env(args) -> bool: ...


def config_from_args(
    args) -> Tuple[LaunchConfig, Union[Callable, str], List[str]]: ...


def run_script_path(training_script: str, *training_script_args: str): ...


def run(args) -> None: ...


def main(args: Incomplete | None = ...) -> None: ...
