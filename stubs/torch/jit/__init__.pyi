# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access,no-name-in-module,undefined-variable


from typing import Any

from _typeshed import Incomplete
from torch._jit_internal import export as export
from torch._jit_internal import Final as Final
from torch._jit_internal import Future as Future
from torch._jit_internal import ignore as ignore
from torch._jit_internal import is_scripting as is_scripting
from torch._jit_internal import unused as unused
from torch.jit._async import fork as fork
from torch.jit._async import wait as wait
from torch.jit._freeze import freeze as freeze
from torch.jit._freeze import optimize_for_inference as optimize_for_inference
from torch.jit._freeze import (
    run_frozen_optimizations as run_frozen_optimizations,
)
from torch.jit._fuser import fuser as fuser
from torch.jit._fuser import (
    last_executed_optimized_graph as last_executed_optimized_graph,
)
from torch.jit._fuser import optimized_execution as optimized_execution
from torch.jit._fuser import set_fusion_strategy as set_fusion_strategy
from torch.jit._script import Attribute as Attribute
from torch.jit._script import CompilationUnit as CompilationUnit
from torch.jit._script import interface as interface
from torch.jit._script import RecursiveScriptClass as RecursiveScriptClass
from torch.jit._script import RecursiveScriptModule as RecursiveScriptModule
from torch.jit._script import script as script
from torch.jit._script import script_method as script_method
from torch.jit._script import ScriptFunction as ScriptFunction
from torch.jit._script import ScriptModule as ScriptModule
from torch.jit._script import ScriptWarning as ScriptWarning
from torch.jit._serialization import (
    jit_module_from_flatbuffer as jit_module_from_flatbuffer,
)
from torch.jit._serialization import load as load
from torch.jit._serialization import save as save
from torch.jit._serialization import (
    save_jit_module_to_flatbuffer as save_jit_module_to_flatbuffer,
)
from torch.jit._trace import is_tracing as is_tracing
from torch.jit._trace import ONNXTracedModule as ONNXTracedModule
from torch.jit._trace import TopLevelTracedModule as TopLevelTracedModule
from torch.jit._trace import trace as trace
from torch.jit._trace import trace_module as trace_module
from torch.jit._trace import TracedModule as TracedModule
from torch.jit._trace import TracerWarning as TracerWarning
from torch.jit._trace import TracingCheckError as TracingCheckError
from torch.utils import set_module as set_module


def export_opnames(m): ...


Error: Incomplete


def annotate(the_type, the_value): ...


def script_if_tracing(fn): ...


def isinstance(obj, target_type): ...


class strict_fusion:
    def __init__(self) -> None: ...
    def __enter__(self) -> None: ...
    def __exit__(self, type: Any, value: Any, tb: Any) -> None: ...


def enable_onednn_fusion(enabled: bool): ...


def onednn_fusion_enabled(): ...
