# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access,no-name-in-module,undefined-variable


from collections.abc import Generator
from typing import List, Tuple

from _typeshed import Incomplete


def optimized_execution(should_optimize) -> Generator[None, None, None]: ...


def fuser(name) -> Generator[None, None, None]: ...


last_executed_optimized_graph: Incomplete


def set_fusion_strategy(strategy: List[Tuple[str, int]]): ...
