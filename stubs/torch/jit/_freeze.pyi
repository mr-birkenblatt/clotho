# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from typing import List, Optional

from torch.jit._script import RecursiveScriptModule as RecursiveScriptModule
from torch.jit._script import ScriptModule as ScriptModule


def freeze(
    mod, preserved_attrs: Optional[List[str]] = ...,
        optimize_numerics: bool = ...): ...


def run_frozen_optimizations(
    mod, optimize_numerics: bool = ...,
        preserved_methods: Optional[List[str]] = ...): ...


def optimize_for_inference(
    mod: ScriptModule,
        other_methods: Optional[List[str]] = ...) -> ScriptModule: ...
