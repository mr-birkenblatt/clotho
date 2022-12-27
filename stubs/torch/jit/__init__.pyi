# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from _typeshed import Incomplete
from torch._jit_internal import Final as Final
from torch._jit_internal import Future as Future


        export as export, ignore as ignore, is_scripting as is_scripting,
        unused as unused
from torch.jit._async import fork as fork
from torch.jit._async import wait as wait
from torch.jit._freeze import freeze as freeze


        optimize_for_inference as optimize_for_inference,
        run_frozen_optimizations as run_frozen_optimizations
from torch.jit._fuser import fuser as fuser


        last_executed_optimized_graph as last_executed_optimized_graph,
        optimized_execution as optimized_execution,
        set_fusion_strategy as set_fusion_strategy
from torch.jit._script import Attribute as Attribute


        CompilationUnit as CompilationUnit,
        RecursiveScriptClass as RecursiveScriptClass,
        RecursiveScriptModule as RecursiveScriptModule,
        ScriptFunction as ScriptFunction, ScriptModule as ScriptModule,
        ScriptWarning as ScriptWarning, interface as interface,
        script as script, script_method as script_method
from torch.jit._serialization import as, jit_module_from_flatbuffer


        jit_module_from_flatbuffer, load as load, save as save,
        save_jit_module_to_flatbuffer as save_jit_module_to_flatbuffer
from torch.jit._trace import ONNXTracedModule as ONNXTracedModule


        TopLevelTracedModule as TopLevelTracedModule,
        TracedModule as TracedModule, TracerWarning as TracerWarning,
        TracingCheckError as TracingCheckError, is_tracing as is_tracing,
        trace as trace, trace_module as trace_module
from typing import Any

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
