# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from _typeshed import Incomplete


def merge(*dicts, **kwargs): ...


def merge_with(func, *dicts, **kwargs): ...


def valmap(func, d, factory=...): ...


def keymap(func, d, factory=...): ...


def itemmap(func, d, factory=...): ...


def valfilter(predicate, d, factory=...): ...


def keyfilter(predicate, d, factory=...): ...


def itemfilter(predicate, d, factory=...): ...


def assoc(d, key, value, factory=...): ...


def dissoc(d, *keys, **kwargs): ...


def assoc_in(d, keys, value, factory=...): ...


def update_in(
    d, keys, func, default: Incomplete | None = ..., factory=...): ...


def get_in(
    keys, coll, default: Incomplete | None = ..., no_default: bool = ...): ...
