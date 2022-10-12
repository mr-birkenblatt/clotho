from typing import (
    Any,
    Callable,
    Generic,
    List,
    Optional,
    TYPE_CHECKING,
    TypeVar,
    Union,
)

from misc.redis import RedisConnection, RedisFunctionBytes
from misc.util import json_compact


if TYPE_CHECKING:
    from effects.redis import SetRootRedisType, ValueRootRedisType


JSONType = Union[str, int, float, List[str], List[int], List[float]]
LiteralType = Union[str, int, float, bool]


KT = TypeVar('KT')
VT = TypeVar('VT')
PT = TypeVar('PT')


class Compilable:  # pylint: disable=too-few-public-methods
    def compile(self, indent: int) -> str:
        raise NotImplementedError()


class Stmt(Compilable):  # pylint: disable=too-few-public-methods
    def __init__(self, code_fn: Callable[[], str]) -> None:
        self._code_fn = code_fn

    def compile(self, indent: int) -> str:
        ind = " " * indent
        code = self._code_fn()
        return "\n".join((f"{ind}{exe}" for exe in code.split("\n")))


class Expr:
    def compile(self) -> str:
        raise NotImplementedError()

    def as_stmt(self) -> Stmt:
        return Stmt(self.compile)


class Sequence(Compilable):
    def __init__(self) -> None:
        self._seq: List[Compilable] = []

    def add_stmt(self, statement: Compilable) -> 'Sequence':
        self._seq.append(statement)
        return self

    def is_empty(self) -> bool:
        return not self._seq

    def compile(self, indent: int) -> str:
        return "\n".join((stmt.compile(indent) for stmt in self._seq))


class Script(Compilable):
    def __init__(self) -> None:
        self._args: List[Arg] = []
        self._keys: List[KeyVariable] = []
        self._locals: List[LocalVariable] = []
        self._return: Optional[Expr] = None
        self._seq = Sequence()
        self._compute: Optional[RedisFunctionBytes] = None
        self._loops: int = 0

    def add_stmt(self, statement: Compilable) -> None:
        self._seq.add_stmt(statement)

    def add_arg(self, arg: 'Arg') -> None:
        self._args.append(arg)
        arg.set_index(len(self._args))

    def add_key(self, key: 'KeyVariable') -> None:
        self._keys.append(key)
        key.set_index(len(self._keys))

    def add_local(self, local: 'LocalVariable') -> None:
        self._locals.append(local)
        local.set_index(len(self._locals))

    def add_loop(self) -> int:
        self._loops += 1
        return self._loops

    def set_return_value(self, expr: Expr) -> None:
        self._return = expr

    def compile(self, indent: int) -> str:
        ind = " " * indent
        keys = "\n".join((f"{ind}{key.get_declare()}" for key in self._keys))
        decl = "\n".join((f"{ind}{arg.get_declare()}" for arg in self._args))
        lcl = "\n".join((f"{ind}{lcl.get_declare()}" for lcl in self._locals))
        if self._return is None:
            ret = ""
        else:
            ret = f"\n{ind}return {self._return.compile()}"
        return f"{keys}\n{decl}\n{lcl}\n{self._seq.compile(indent)}{ret}\n"

    def execute(
            self,
            *,
            args: List[JSONType],
            keys: List[Any],
            conn: RedisConnection) -> bytes:
        if self._compute is None:
            code = self.compile(0)
            self._compute = conn.get_dynamic_script(code)
        with conn.get_connection() as client:
            res = self._compute(
                keys=[
                    key_var.get_value(key)
                    for (key, key_var) in zip(keys, self._keys)
                ],
                args=[
                    arg_var.get_value(arg)
                    for (arg, arg_var) in zip(args, self._args)
                ],
                client=client)
        for (key, key_var) in zip(keys, self._keys):
            key_var.post_completion(key)
        return res


class Variable(Expr):
    def __init__(self) -> None:
        self._index: Optional[int] = None

    def set_index(self, index: int) -> None:
        self._index = index

    def get_index(self) -> int:
        assert self._index is not None
        return self._index

    def prefix(self) -> str:
        raise NotImplementedError()

    def assign(self, expr: Expr) -> Stmt:
        return Stmt(lambda: f"{self.get_ref()} = {expr.compile()}")

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

    def post_completion(self, key: KT) -> None:
        raise NotImplementedError()


class LiteralKey(KeyVariable[str]):
    def get_value(self, key: str) -> str:
        return key

    def post_completion(self, key: str) -> None:
        pass


class LocalVariable(Variable):
    def __init__(self, init: Expr) -> None:
        super().__init__()
        self._init = init

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
        res = res.replace("\"", "\\\"")
        return f"\"{res}\""


class Op(Expr):  # pylint: disable=too-few-public-methods
    def __init__(self, lhs: Expr, rhs: Expr) -> None:
        self._lhs = lhs
        self._rhs = rhs

    def get_left(self) -> str:
        return self._lhs.compile()

    def get_right(self) -> str:
        return self._rhs.compile()

    def compile(self) -> str:
        raise NotImplementedError()


class AddOp(Op):  # pylint: disable=too-few-public-methods
    def compile(self) -> str:
        return f"({self.get_left()} + {self.get_right()})"


class LtOp(Op):  # pylint: disable=too-few-public-methods
    def compile(self) -> str:
        return f"({self.get_left()} < {self.get_right()})"


class EqOp(Op):  # pylint: disable=too-few-public-methods
    def compile(self) -> str:
        return f"({self.get_left()} == {self.get_right()})"


class CallFn(Expr):  # pylint: disable=too-few-public-methods
    def __init__(self, fname: str, *args: Expr) -> None:
        self._fname = fname
        self._args = args

    def compile(self) -> str:
        argstr = ", ".join((arg.compile() for arg in self._args))
        return f"{self._fname}({argstr})"


class Branch(Compilable):
    def __init__(self, condition: Expr) -> None:
        self._condition = condition
        self._success = Sequence()
        self._failure = Sequence()

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
        self._loop = Sequence()
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


# user_id = who.get_id()
# if store.r_voted.add_value(key, user_id):
#     return
# votes = self.get_votes(vote_type)
# weighted_value = who.get_weighted_vote(self.get_user(user_store))
# store.r_total.set_value(key, votes.get_total_votes() + weighted_value)
# store.r_daily.set_value(
#     key, votes.get_adjusted_daily_votes(now) + weighted_value)
# store.r_user.do_set_new_value(key, user_id)
# nows = to_timestamp(now)
# is_new = store.r_first.set_new_value(key, nows)
# store.r_last.set_value(key, nows)
# if is_new and key.vote_type == VT_UP:
#     store.r_user_links.add_value(
#         user_id, parseable_link(key.parent, key.child))


class RootValue(Generic[KT, VT], KeyVariable[KT]):
    def __init__(self, ref: 'ValueRootRedisType[KT, VT]') -> None:
        self._ref = ref

    def get_value(self, key: KT) -> str:
        return self._ref.get_redis_key(key)

    def post_completion(self, key: KT) -> None:
        self._ref.on_update(key)


class RootSet(Generic[KT, VT], KeyVariable[KT]):
    def __init__(self, ref: 'SetRootRedisType[KT]') -> None:
        self._ref = ref

    def get_value(self, key: KT) -> str:
        return self._ref.get_redis_key(key)

    def post_completion(self, key: KT) -> None:
        self._ref.on_update(key)
