# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from .utils import groupby as groupby
from .variadic import isvariadic as isvariadic


class AmbiguityWarning(Warning):
    ...


def supercedes(a, b): ...


def consistent(a, b): ...


def ambiguous(a, b): ...


def ambiguities(signatures): ...


def super_signature(signatures): ...


def edge(a, b, tie_breaker=...): ...


def ordering(signatures): ...
