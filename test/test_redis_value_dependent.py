import time
from typing import List, Tuple

from effects.effects import EffectDependent
from effects.redis import ValueDependentRedisType, ValueRootRedisType


def test_value() -> None:
    value_a: ValueRootRedisType[str, int] = ValueRootRedisType(
        "test", lambda key: f"count:{key}")
    value_b: ValueRootRedisType[str, str] = ValueRootRedisType(
        "test", lambda key: f"name:{key}")

    def update_a(
            obj: EffectDependent[str, List[str]],
            parents: Tuple[
                ValueRootRedisType[str, int], ValueRootRedisType[str, str]],
            key: str) -> None:
        v_a, v_b = parents
        obj.set_value(
            key, [v_b.get_value(key, "MISSING")] * v_a.get_value(key, 0))

    def update_b(
            obj: EffectDependent[str, int],
            parents: Tuple[ValueRootRedisType[str, str]],
            key: str) -> None:
        v_a, = parents
        obj.set_value(key, len(v_a.get_value(key, "MISSING")))

    dep_a: ValueDependentRedisType[str, List[str]] = ValueDependentRedisType(
        "test",
        lambda key: f"list:{key}",
        (value_a, value_b),
        update_a,
        0.1)
    dep_b: ValueDependentRedisType[str, int] = ValueDependentRedisType(
        "test",
        lambda key: f"len:{key}",
        (value_b,),
        update_b,
        0.1)

    assert value_a.maybe_get_value("a") is None
    value_a.set_value("a", 2)
    assert value_a.maybe_get_value("a") == 2
    time.sleep(0.2)
    assert dep_a.maybe_get_value("a") == ["MISSING", "MISSING"]
    assert dep_b.maybe_get_value("a") == 7

    value_a.set_value("a", 3)
    value_b.set_value("a", "abc")
    value_a.set_value("b", 2)
    value_b.set_value("b", "defg")
    time.sleep(0.2)
    assert dep_a.maybe_get_value("a") == ["abc", "abc", "abc",]
    assert dep_b.maybe_get_value("a") == 3
    assert dep_a.maybe_get_value("b") == ["defg", "defg"]
    assert dep_b.maybe_get_value("b") == 4

    def update_a_a(
            obj: EffectDependent[str, str],
            parents: Tuple[ValueDependentRedisType[str, List[str]]],
            key: str) -> None:
        v_a, = parents
        obj.set_value(key, "-".join(v_a.get_value(key, [])))

    dep_a_a: ValueDependentRedisType[str, str] = ValueDependentRedisType(
        "test",
        lambda key: f"concat:{key}",
        (dep_a,),
        update_a_a,
        0.1)

    assert dep_a_a.maybe_get_value("a") == "abc-abc-abc"

    value_a.set_value("a", 5)
    value_b.set_value("a", "+")
    time.sleep(0.4)
    assert dep_a_a.maybe_get_value("a") == "+-+-+-+-+"
    assert dep_a.maybe_get_value("a") == ["+", "+", "+", "+", "+"]
    assert dep_b.maybe_get_value("a") == 1
    assert value_a.maybe_get_value("a") == 5
    assert value_b.maybe_get_value("a") == "+"
    assert value_a.maybe_get_value("b") == 2
    assert value_b.maybe_get_value("b") == "defg"
    assert dep_a_a.maybe_get_value("b") == "defg-defg"
    assert dep_a.maybe_get_value("b") == ["defg", "defg"]
    assert dep_b.maybe_get_value("b") == 4
