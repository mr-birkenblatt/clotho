# pylint: disable=multiple-statements,unused-argument, invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias, unused-import
# pylint: disable=redefined-builtin,super-init-not-called, arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors, import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member, keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name, c-extension-no-member
# pylint: disable=protected-access,no-name-in-module, undefined-variable


from typing import Dict

import torch


def _get_module_info_from_flatbuffer(arg0: str) -> dict: ...


def _load_jit_module_from_bytes(arg0: str) -> torch.ScriptModule: ...


def _load_jit_module_from_file(arg0: str) -> torch.ScriptModule: ...


def _load_mobile_module_from_bytes(arg0: str) -> torch.LiteScriptModule: ...


def _load_mobile_module_from_file(arg0: str) -> torch.LiteScriptModule: ...


def _save_jit_module(
    arg0: torch.ScriptModule, arg1: str, arg2: Dict[str, str]) -> None: ...


def _save_jit_module_to_bytes(
    arg0: torch.ScriptModule, arg1: Dict[str, str]) -> bytes: ...


def _save_mobile_module(
    arg0: torch.LiteScriptModule, arg1: str, arg2: Dict[str, str]) -> None: ...


def _save_mobile_module_to_bytes(
    arg0: torch.LiteScriptModule, arg1: Dict[str, str]) -> bytes: ...
