# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access,no-name-in-module,undefined-variable


from _typeshed import Incomplete
from torch._six import string_classes as string_classes
from torch.jit._recursive import wrap_cpp_module as wrap_cpp_module
from torch.serialization import validate_cuda_device as validate_cuda_device


def save(m, f, _extra_files: Incomplete | None = ...) -> None: ...


def load(
    f, map_location: Incomplete | None = ...,
    _extra_files: Incomplete | None = ...): ...


def validate_map_location(map_location: Incomplete | None = ...): ...


def get_ff_module(): ...


def jit_module_from_flatbuffer(f): ...


def save_jit_module_to_flatbuffer(
    m, f, _extra_files: Incomplete | None = ...) -> None: ...


def get_flatbuffer_module_info(path_or_file): ...
