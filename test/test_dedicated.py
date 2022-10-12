from effects.dedicated import (
    AddOp,
    Arg,
    Branch,
    CallFn,
    EqOp,
    Literal,
    LocalVariable,
    LtOp,
    RootValue,
    Script,
)
from effects.redis import ValueRootRedisType
from misc.redis import RedisConnection


SCRIPT_REF = """
local key_1 = KEYS[1]
local arg_1 = cjson.decode(ARGV[1])
local arg_2 = cjson.decode(ARGV[2])
local var_1 = 0.0
local var_2 = key_1
if (arg_1 < 1.0) then
    var_1 = (arg_1 + arg_2)
else
    if (string.sub(var_2, -4) == ":abc") then
        var_1 = -1.0
    else
        var_1 = 1.0
    end
end
redis.call("SET", key_1, var_1)
return 1
""".lstrip().replace("    ", "  ")


def test_dedicated() -> None:
    script = Script()
    input_a = script.add_arg(Arg())
    input_b = script.add_arg(Arg())

    value_a: ValueRootRedisType[str, float] = ValueRootRedisType(
        "test", lambda key: key)
    output_a: RootValue[str, float] = script.add_key(RootValue(value_a))
    var_a = script.add_local(LocalVariable(Literal(0.0)))
    var_b = script.add_local(LocalVariable(output_a))

    postfix = CallFn("string.sub", var_b, Literal(-4))
    branch_inner = Branch(EqOp(postfix, Literal(":abc")))
    branch_inner.get_success().add_stmt(var_a.assign(Literal(-1.0)))
    branch_inner.get_failure().add_stmt(var_a.assign(Literal(1.0)))

    branch = Branch(LtOp(input_a, Literal(1.0)))
    branch.get_success().add_stmt(var_a.assign(AddOp(input_a, input_b)))
    branch.get_failure().add_stmt(branch_inner)

    script.add_stmt(branch).add_stmt(
        CallFn("redis.call", Literal("SET"), output_a, var_a).as_stmt())

    script.set_return_value(Literal(1))

    assert script.compile(0) == SCRIPT_REF

    conn = RedisConnection("test")
    assert value_a.maybe_get_value("abc") is None
    assert value_a.maybe_get_value("def") is None
    script.execute(args=[-1.0, 3.0], keys=["def"], conn=conn)
    assert value_a.maybe_get_value("def") == 2.0
    script.execute(args=[3.0, -1.0], keys=["def"], conn=conn)
    assert value_a.maybe_get_value("def") == 1.0
    assert value_a.maybe_get_value("abc") is None

    assert int(script.execute(args=[3.0, -1.0], keys=["abc"], conn=conn)) != 0
    assert value_a.maybe_get_value("abc") == -1.0