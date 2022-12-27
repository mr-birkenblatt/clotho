# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from .utils import typename as typename


class VariadicSignatureType(type):
    def __subclasscheck__(cls, subclass): ...
    def __eq__(cls, other): ...
    def __hash__(cls): ...


def isvariadic(obj): ...


class VariadicSignatureMeta(type):
    def __getitem__(cls, variadic_type): ...


class Variadic:
    ...
