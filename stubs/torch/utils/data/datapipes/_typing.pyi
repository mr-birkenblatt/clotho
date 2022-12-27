# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level
import abc
import numbers
from abc import ABCMeta
from typing import TypeVar

from _typeshed import Incomplete


class GenericMeta(ABCMeta):
    ...


class Integer(numbers.Integral, metaclass=abc.ABCMeta):
    ...


class Boolean(numbers.Integral, metaclass=abc.ABCMeta):
    ...


TYPE2ABC: Incomplete


def issubtype(left, right, recursive: bool = ...): ...


def issubinstance(data, data_type): ...


class _DataPipeType:
    param: Incomplete
    def __init__(self, param) -> None: ...
    def __eq__(self, other): ...
    def __hash__(self): ...
    def issubtype(self, other): ...
    def issubtype_of_instance(self, other): ...


T_co = TypeVar('T_co', covariant=True)


class _DataPipeMeta(GenericMeta):
    type: _DataPipeType
    def __new__(cls, name, bases, namespace, **kwargs): ...
    def __init__(cls, name, bases, namespace, **kwargs) -> None: ...


class _IterDataPipeMeta(_DataPipeMeta):
    def __new__(cls, name, bases, namespace, **kwargs): ...


def hook_iterator(namespace, profile_name): ...


def reinforce_type(self, expected_type): ...
