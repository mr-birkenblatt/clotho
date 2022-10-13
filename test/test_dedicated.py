from effects.dedicated import (
    AddOp,
    CallFn,
    EqOp,
    LtOp,
    RedisFn,
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
    input_a = script.add_arg("input_a")
    input_b = script.add_arg("input_b")

    value_a: ValueRootRedisType[str, float] = ValueRootRedisType(
        "test", lambda key: key)
    output_a: RootValue[str, float] = script.add_key(
        "value_a", RootValue(value_a))
    var_a = script.add_local(0.0)
    var_b = script.add_local(output_a)

    success, failure = script.branch(LtOp(input_a, 1.0))
    success.add(var_a.assign(AddOp(input_a, input_b)))

    inner_success, inner_failure = failure.branch(
        EqOp(CallFn("string.sub", var_b, -4), ":abc"))

    inner_success.add(var_a.assign(-1.0))
    inner_failure.add(var_a.assign(1.0))

    script.add(RedisFn("SET", output_a, var_a))

    script.set_return_value(1)

    assert script.compile(0) == SCRIPT_REF

    conn = RedisConnection("test")
    assert value_a.maybe_get_value("abc") is None
    assert value_a.maybe_get_value("def") is None
    script.execute(
        args={"input_a": -1.0, "input_b": 3.0},
        keys={"value_a": "def"},
        conn=conn)
    assert value_a.maybe_get_value("def") == 2.0
    script.execute(
        args={"input_a": 3.0, "input_b": -1.0},
        keys={"value_a": "def"},
        conn=conn)
    assert value_a.maybe_get_value("def") == 1.0
    assert value_a.maybe_get_value("abc") is None

    assert int(script.execute(
        args={"input_a": 3.0, "input_b": -1.0},
        keys={"value_a": "abc"},
        conn=conn)) != 0
    assert value_a.maybe_get_value("abc") == -1.0
