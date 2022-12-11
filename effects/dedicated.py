from typing import (
    Any,
    Callable,
    Generic,
    Iterable,
    TYPE_CHECKING,
    TypeVar,
    Union,
)

import pandas as pd

from effects.effects import KeyType
from misc.redis import RedisConnection, RedisFunctionBytes
from misc.util import json_compact


if TYPE_CHECKING:
    from effects.redis import SetRootRedisType, ValueRootRedisType


JSONType = str | int | float | list[str] | list[int] | list[float]
LiteralType = str | int | float | bool
MixedType = Union['Expr', LiteralType]


KT = TypeVar('KT', bound=KeyType)
VT = TypeVar('VT')
PT = TypeVar('PT', bound=KeyType)
KV = TypeVar('KV', bound='KeyVariable')


class Compilable:  # pylint: disable=too-few-public-methods
    def compile(self, indent: int) -> str:
        raise NotImplementedError()


class ExprHelper(Compilable):  # pylint: disable=too-few-public-methods
    def __init__(self, code_fn: Callable[[], str]) -> None:
        self._code_fn = code_fn

    def compile(self, indent: int) -> str:
        ind = " " * indent
        code = self._code_fn()
        return "\n".join((f"{ind}{exe}" for exe in code.split("\n")))


class Expr:
    def compile(self) -> str:
        raise NotImplementedError()

    def as_stmt(self) -> ExprHelper:
        return ExprHelper(self.compile)

    def __add__(self, other: MixedType) -> 'Expr':
        return AddOp(self, other)

    def __sub__(self, other: MixedType) -> 'Expr':
        return SubOp(self, other)

    def eq(self, other: MixedType) -> 'Expr':  # pylint: disable=invalid-name
        return EqOp(self, other)

    def neq(self, other: MixedType) -> 'Expr':
        return NeqOp(self, other)

    def lt(self, other: MixedType) -> 'Expr':  # pylint: disable=invalid-name
        return LtOp(self, other)

    def not_(self) -> 'Expr':
        return NotOp(self)

    def or_(self, other: MixedType) -> 'Expr':
        return OrOp(self, other)

    def and_(self, other: MixedType) -> 'Expr':
        return AndOp(self, other)

    def json(self) -> 'Expr':
        return ToJSON(self)


def lit_helper(value: MixedType) -> 'Expr':
    if isinstance(value, Expr):
        return value
    return Literal(value)


class Sequence(Compilable):
    def __init__(self, script: 'Script') -> None:
        self._seq: list[Compilable] = []
        self._script = script

    def add(
            self,
            terms: Compilable | Expr | Iterable[Compilable | Expr]) -> None:
        if isinstance(terms, Compilable):
            self._seq.append(terms)
        elif isinstance(terms, Expr):
            self._seq.append(terms.as_stmt())
        else:
            for term in terms:
                self.add(term)

    def is_empty(self) -> bool:
        return not self._seq

    def compile(self, indent: int) -> str:
        return "\n".join((stmt.compile(indent) for stmt in self._seq))

    def for_(self, array: Expr) -> tuple['Sequence', 'Variable', 'Variable']:
        loop = ForLoop(self._script, array)
        self.add(loop)
        return loop.get_loop(), loop.get_index(), loop.get_value()

    def if_(self, condition: MixedType) -> tuple['Sequence', 'Sequence']:
        branch = Branch(self._script, condition)
        self.add(branch)
        return branch.get_success(), branch.get_failure()


class Script(Sequence):
    def __init__(self) -> None:
        super().__init__(self)
        self._args: list[tuple[str, Arg]] = []
        self._keys: list[tuple[str, KeyVariable]] = []
        self._anames: set[str] = set()
        self._knames: set[str] = set()
        self._locals: list[LocalVariable] = []
        self._return: Expr | None = None
        self._compute: RedisFunctionBytes | None = None
        self._loops: int = 0

    def add_arg(self, name: str) -> 'Arg':
        if name in self._anames:
            raise ValueError(f"ambiguous arg name: '{name}'")
        arg = Arg()
        self._args.append((name, arg))
        self._anames.add(name)
        arg.set_index(len(self._args))
        return arg

    def add_key(self, name: str, key: KV) -> KV:
        if name in self._knames:
            raise ValueError(f"ambiguous key name: '{name}'")
        self._keys.append((name, key))
        self._knames.add(name)
        key.set_index(len(self._keys))
        return key

    def add_local(self, init: MixedType) -> 'LocalVariable':
        local = LocalVariable(init)
        self._locals.append(local)
        local.set_index(len(self._locals))
        return local

    def add_loop(self) -> int:
        self._loops += 1
        return self._loops

    def set_return_value(self, expr: MixedType) -> None:
        self._return = lit_helper(expr)

    def compile(self, indent: int) -> str:
        ind = " " * indent
        keys = "\n".join(
            (f"{ind}{key.get_declare()}" for _, key in self._keys))
        decl = "\n".join(
            (f"{ind}{arg.get_declare()}" for _, arg in self._args))
        lcl = "\n".join((f"{ind}{lcl.get_declare()}" for lcl in self._locals))
        if self._return is None:
            ret = ""
        else:
            ret = f"\n{ind}return {self._return.compile()}"
        return f"{keys}\n{decl}\n{lcl}\n{super().compile(indent)}{ret}\n"

    def execute(
            self,
            *,
            args: dict[str, JSONType],
            keys: dict[str, Any],
            now: pd.Timestamp,
            conn: RedisConnection,
            depth: int) -> bytes:
        assert len(keys) == len(self._keys)
        assert len(args) == len(self._args)
        argv = [
            arg_var.get_value(args[aname])
            for aname, arg_var in self._args
        ]
        keyv = [
            key_var.get_value(keys[kname])
            for kname, key_var in self._keys
        ]
        if self._compute is None:
            code = self.compile(0)
            self._compute = conn.get_dynamic_script(code)
        with conn.get_connection(depth=depth + 1) as client:
            res = self._compute(
                keys=keyv, args=argv, client=client, depth=depth + 1)
        for kname, key_var in self._keys:
            key_var.post_completion(keys[kname], now)
        return res


class Variable(Expr):
    def __init__(self) -> None:
        self._index: int | None = None

    def set_index(self, index: int) -> None:
        self._index = index

    def get_index(self) -> int:
        assert self._index is not None
        return self._index

    def prefix(self) -> str:
        raise NotImplementedError()

    def assign(self, expr: MixedType) -> ExprHelper:
        return ExprHelper(
            lambda: f"{self.get_ref()} = {lit_helper(expr).compile()}")

    def get_declare(self) -> str:
        return f"local {self.get_ref()} = {self.get_declare_rhs()}"

    def get_declare_rhs(self) -> str:
        raise NotImplementedError()

    def get_ref(self) -> str:
        return f"{self.prefix()}_{self.get_index()}"

    def compile(self) -> str:
        return self.get_ref()


class Arg(Variable):
    def get_declare_rhs(self) -> str:
        return f"cjson.decode(ARGV[{self.get_index()}])"

    def prefix(self) -> str:
        return "arg"

    def get_value(self, value: JSONType) -> bytes:
        return json_compact(value)


class KeyVariable(Generic[KT], Variable):
    def get_declare_rhs(self) -> str:
        return f"KEYS[{self.get_index()}]"

    def prefix(self) -> str:
        return "key"

    def get_value(self, key: KT) -> str:
        raise NotImplementedError()

    def post_completion(self, key: KT, now: pd.Timestamp) -> None:
        raise NotImplementedError()


class LiteralKey(KeyVariable[str]):
    def get_value(self, key: str) -> str:
        return key

    def post_completion(self, key: str, now: pd.Timestamp) -> None:
        pass


class LocalVariable(Variable):
    def __init__(self, init: MixedType) -> None:
        super().__init__()
        self._init = lit_helper(init)

    def get_declare_rhs(self) -> str:
        return self._init.compile()

    def prefix(self) -> str:
        return "var"


class Literal(Expr):  # pylint: disable=too-few-public-methods
    def __init__(self, value: LiteralType) -> None:
        self._value = value

    def compile(self) -> str:
        if isinstance(self._value, bool):
            return f"{self._value}".lower()
        if isinstance(self._value, (int, float)):
            return f"{self._value}"
        res = f"{self._value}"
        res = res.replace("\"", "\\\"").replace("\n", "\\n")
        return f"\"{res}\""


class NotOp(Expr):  # pylint: disable=too-few-public-methods
    def __init__(self, expr: MixedType) -> None:
        self._expr = lit_helper(expr)

    def compile(self) -> str:
        return f"(not {self._expr.compile()})"


class Op(Expr):
    def __init__(self, lhs: MixedType, rhs: MixedType) -> None:
        self._lhs = lit_helper(lhs)
        self._rhs = lit_helper(rhs)

    def get_left(self) -> str:
        return self._lhs.compile()

    def get_right(self) -> str:
        return self._rhs.compile()

    def compile(self) -> str:
        raise NotImplementedError()


class AndOp(Op):  # pylint: disable=too-few-public-methods
    def compile(self) -> str:
        return f"({self.get_left()} and {self.get_right()})"


class OrOp(Op):  # pylint: disable=too-few-public-methods
    def compile(self) -> str:
        return f"({self.get_left()} or {self.get_right()})"


class AddOp(Op):  # pylint: disable=too-few-public-methods
    def compile(self) -> str:
        return f"({self.get_left()} + {self.get_right()})"


class SubOp(Op):  # pylint: disable=too-few-public-methods
    def compile(self) -> str:
        return f"({self.get_left()} - {self.get_right()})"


class LtOp(Op):  # pylint: disable=too-few-public-methods
    def compile(self) -> str:
        return f"({self.get_left()} < {self.get_right()})"


class EqOp(Op):  # pylint: disable=too-few-public-methods
    def compile(self) -> str:
        return f"({self.get_left()} == {self.get_right()})"


class NeqOp(Op):  # pylint: disable=too-few-public-methods
    def compile(self) -> str:
        return f"({self.get_left()} ~= {self.get_right()})"


class CallFn(Expr):  # pylint: disable=too-few-public-methods
    def __init__(self, fname: str, *args: MixedType) -> None:
        self._fname = fname
        self._args: list[Expr] = [lit_helper(arg) for arg in args]

    def compile(self) -> str:
        argstr = ", ".join((arg.compile() for arg in self._args))
        return f"{self._fname}({argstr})"


class ToJSON(CallFn):  # pylint: disable=too-few-public-methods
    def __init__(self, arg: MixedType) -> None:
        super().__init__("cjson.encode", arg)


class RedisFn(CallFn):  # pylint: disable=too-few-public-methods
    def __init__(
            self, redis_fn: str, key: KeyVariable, *args: MixedType) -> None:
        super().__init__("redis.call", redis_fn, key, *args)


class Branch(Compilable):
    def __init__(self, script: Script, condition: MixedType) -> None:
        self._condition = lit_helper(condition)
        self._success = Sequence(script)
        self._failure = Sequence(script)

    def get_success(self) -> Sequence:
        return self._success

    def get_failure(self) -> Sequence:
        return self._failure

    def compile(self, indent: int) -> str:
        ind = indent * " "
        start = f"{ind}if {self._condition.compile()} then\n"
        block_a = f"{self._success.compile(indent + 2)}\n"
        if self._failure.is_empty():
            middle = ""
            block_b = ""
        else:
            middle = f"{ind}else\n"
            block_b = f"{self._failure.compile(indent + 2)}\n"
        end = f"{ind}end"
        return f"{start}{block_a}{middle}{block_b}{end}"


class IndexVariable(Variable):
    def get_declare_rhs(self) -> str:
        raise RuntimeError("must be used in for loop")

    def prefix(self) -> str:
        return "ix"


class ValueVariable(Variable):
    def get_declare_rhs(self) -> str:
        raise RuntimeError("must be used in for loop")

    def prefix(self) -> str:
        return "val"


class ForLoop(Compilable):
    def __init__(self, script: Script, array: Expr) -> None:
        loop_ix = script.add_loop()
        self._ix = IndexVariable()
        self._ix.set_index(loop_ix)
        self._val = ValueVariable()
        self._val.set_index(loop_ix)
        self._loop = Sequence(script)
        self._array = array

    def get_index(self) -> Variable:
        return self._ix

    def get_value(self) -> Variable:
        return self._val

    def get_loop(self) -> Sequence:
        return self._loop

    def compile(self, indent: int) -> str:
        ind = indent * " "
        start_a = f"{ind}for {self._ix.get_ref()}, {self._val.get_ref()} "
        start_b = f"in pairs({self._array.compile()}) do\n"
        block = f"{self._loop.compile(indent + 2)}\n"
        end = f"{ind}end"
        return f"{start_a}{start_b}{block}{end}"


class RootValue(Generic[KT, VT], KeyVariable[KT]):
    def __init__(self, ref: 'ValueRootRedisType[KT, VT]') -> None:
        self._ref = ref

    def get_value(self, key: KT) -> str:
        return self._ref.get_redis_key(key)

    def post_completion(self, key: KT, now: pd.Timestamp) -> None:
        self._ref.on_update(key, now)


class RootSet(Generic[KT], KeyVariable[KT]):
    def __init__(self, ref: 'SetRootRedisType[KT]') -> None:
        self._ref = ref

    def get_value(self, key: KT) -> str:
        return self._ref.get_redis_key(key)

    def post_completion(self, key: KT, now: pd.Timestamp) -> None:
        self._ref.on_update(key, now)
