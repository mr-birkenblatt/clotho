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
