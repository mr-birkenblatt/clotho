from torch._six import with_metaclass as with_metaclass


class VariableMeta(type):
    def __instancecheck__(cls, other): ...

class Variable: ...
