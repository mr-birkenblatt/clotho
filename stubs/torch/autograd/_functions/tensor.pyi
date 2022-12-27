# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from ..function import Function as Function


class Type(Function):
    @staticmethod
    def forward(ctx, i, dest_type): ...
    @staticmethod
    def backward(ctx, grad_output): ...


class Resize(Function):
    @staticmethod
    def forward(ctx, tensor, sizes): ...
    @staticmethod
    def backward(ctx, grad_output): ...
