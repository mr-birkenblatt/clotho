# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access,no-name-in-module,undefined-variable


from _typeshed import Incomplete


def vjp(
    func, inputs, v: Incomplete | None = ..., create_graph: bool = ...,
    strict: bool = ...): ...


def jvp(
    func, inputs, v: Incomplete | None = ..., create_graph: bool = ...,
    strict: bool = ...): ...


def jacobian(
    func, inputs, create_graph: bool = ..., strict: bool = ...,
    vectorize: bool = ..., strategy: str = ...): ...


def hessian(
    func, inputs, create_graph: bool = ..., strict: bool = ...,
    vectorize: bool = ..., outer_jacobian_strategy: str = ...): ...


def vhp(
    func, inputs, v: Incomplete | None = ..., create_graph: bool = ...,
    strict: bool = ...): ...


def hvp(
    func, inputs, v: Incomplete | None = ..., create_graph: bool = ...,
    strict: bool = ...): ...
