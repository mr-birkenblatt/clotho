import time

from effects.redis import (
    ListDependentRedisType,
    ValueDependentRedisType,
    ValueRootRedisType,
)


def test_dependent() -> None:
    value_a: ValueRootRedisType[str, int] = ValueRootRedisType(
        "test", lambda key: f"count:{key}")
    value_b: ValueRootRedisType[str, str] = ValueRootRedisType(
        "test", lambda key: f"name:{key}")

    def key_id(key: str) -> str:
        return key

    def update_a(key: str) -> None:
        dep_a.set_value(
            key,
            [value_b.get_value(key, "MISSING")] * value_a.get_value(key, 0))

    def update_b(key: str) -> None:
        ref = dep_b.retrieve_value(key)
        old = dep_b.update_value(key, len(value_b.get_value(key, "MISSING")))
        assert (old is None and ref is None) or old == ref

    dep_a: ValueDependentRedisType[str, list[str]] = \
        ValueDependentRedisType(
            "test",
            lambda key: f"list:{key}",
            parents=(value_a, value_b),
            convert=key_id,
            effect=update_a,
            delay=0.1)
    dep_b: ValueDependentRedisType[str, int] = \
        ValueDependentRedisType(
            "test",
            lambda key: f"len:{key}",
            parents=(value_b,),
            convert=key_id,
            effect=update_b,
            delay=0.1)

    assert value_a.maybe_get_value("a") is None
    value_a.set_value("a", 2)
    assert value_a.maybe_get_value("a") == 2
    value_b.on_update("a")
    time.sleep(0.2)
    assert dep_a.maybe_get_value("a") == ["MISSING", "MISSING"]
    assert dep_b.maybe_get_value("a") == 7

    value_a.set_value("a", 3)
    value_b.set_value("a", "abc")
    value_a.set_value("b", 2)
    value_b.set_value("b", "defg")
    time.sleep(0.2)
    assert dep_a.maybe_get_value("a") == ["abc", "abc", "abc"]
    assert dep_b.maybe_get_value("a") == 3
    assert dep_a.maybe_get_value("b") == ["defg", "defg"]
    assert dep_b.maybe_get_value("b") == 4

    def update_a_a(key: str) -> None:
        dep_a_a.set_value(key, "-".join(dep_a.get_value(key, [])))

    def update_a_b(key: str) -> None:
        new = "-".join(dep_a.get_value(key, []))
        old = dep_a_b.maybe_get_value(key)
        if dep_a_b.set_new_value(key, new):
            assert dep_a_b.maybe_get_value(key) == new
        else:
            assert dep_a_b.maybe_get_value(key) == old

    def update_a_c(key: str) -> None:
        val = dep_a.get_value(key, [])
        if val:
            dep_a_c.set_value(key, "-".join(val))

    dep_a_a: ValueDependentRedisType[str, str] = ValueDependentRedisType(
        "test",
        lambda key: f"concat:{key}",
        parents=(dep_a,),
        convert=key_id,
        effect=update_a_a,
        delay=0.1)
    dep_a_b: ValueDependentRedisType[str, str] = ValueDependentRedisType(
        "test",
        lambda key: f"first:{key}",
        parents=(dep_a,),
        convert=key_id,
        effect=update_a_b,
        delay=0.1)
    dep_a_c: ValueDependentRedisType[str, str] = ValueDependentRedisType(
        "test",
        lambda key: f"never:{key}",
        parents=(dep_a,),
        convert=key_id,
        effect=update_a_c,
        delay=0.1)

    dep_a.on_update("a")
    dep_a.on_update("b")
    dep_a.on_update("c")
    time.sleep(0.2)

    assert dep_a_a.maybe_get_value("a") == "abc-abc-abc"
    assert dep_a_b.maybe_get_value("a") == "abc-abc-abc"
    assert dep_a_c.maybe_get_value("a") == "abc-abc-abc"

    value_a.set_value("a", 7)
    value_b.set_value("a", "=")
    value_a.set_value("a", 5)
    value_b.set_value("a", "+")
    assert dep_a_a.maybe_get_value("a") == "abc-abc-abc"  # time sensitive
    time.sleep(0.4)
    assert dep_a_a.maybe_get_value("a") == "+-+-+-+-+"
    assert dep_a_b.maybe_get_value("a") == "abc-abc-abc"
    assert dep_a.maybe_get_value("a") == ["+", "+", "+", "+", "+"]
    assert dep_b.maybe_get_value("a") == 1
    assert value_a.maybe_get_value("a") == 5
    assert value_b.maybe_get_value("a") == "+"
    assert value_a.maybe_get_value("b") == 2
    assert value_b.maybe_get_value("b") == "defg"
    assert dep_a_a.maybe_get_value("b") == "defg-defg"
    assert dep_a.maybe_get_value("b") == ["defg", "defg"]
    assert dep_b.maybe_get_value("b") == 4

    assert dep_a_a.get_value("c", "MISSING") == ""
    assert dep_a_b.maybe_get_value("c") == ""
    assert dep_a_c.get_value("c", "MISSING") == "MISSING"


def test_dependent_list() -> None:
    value_a: ValueRootRedisType[str, int] = ValueRootRedisType(
        "test", lambda key: f"count:{key}")
    value_b: ValueRootRedisType[str, str] = ValueRootRedisType(
        "test", lambda key: f"name:{key}")

    def key_id(key: str) -> str:
        return key

    def update_a(key: str) -> None:
        ref = dep_a.retrieve_value(key)
        old = dep_a.update_value(
            key,
            [value_b.get_value(key, "MISSING")] * value_a.get_value(key, 0))
        assert (old == [] and ref is None) or old == ref

    def update_b(key: str) -> None:
        arr = [value_b.get_value(key, "MISSING")] * value_a.get_value(key, 0)
        old = dep_b.maybe_get_value(key)
        if dep_b.set_new_value(key, arr):
            assert dep_b.maybe_get_value(key) == arr
        else:
            assert dep_b.maybe_get_value(key) == old

    dep_a = ListDependentRedisType(
        "test",
        lambda key: f"slista:{key}",
        parents=(value_a, value_b),
        convert=key_id,
        effect=update_a,
        delay=0.1)
    dep_b = ListDependentRedisType(
        "test",
        lambda key: f"slistb:{key}",
        parents=(value_a, value_b),
        convert=key_id,
        effect=update_b,
        delay=0.1)

    def update_a_a(key: str) -> None:
        dep_a_a.set_value(key, len(dep_a.get_value(key, [])))

    dep_a_a: ValueDependentRedisType[str, int] = ValueDependentRedisType(
        "test",
        lambda key: f"counta:{key}",
        parents=(dep_a,),
        convert=key_id,
        effect=update_a_a,
        delay=0.1)

    value_a.set_value("a", 5)
    value_b.set_value("a", ".")
    value_a.set_value("b", 3)
    value_b.set_value("b", ":")

    assert dep_a_a.maybe_get_value("b") is None  # time sensitive
    assert dep_a.maybe_get_value("b") is None  # time sensitive
    assert dep_b.maybe_get_value("b") is None  # time sensitive
    dep_b.settle_all()

    value_a.set_value("a", 3)

    assert dep_a_a.maybe_get_value("a") is None  # time sensitive
    assert dep_b.maybe_get_value("a") == [".", ".", ".", ".", "."]

    dep_a_a.settle_all()

    assert dep_b.maybe_get_value("a") == [".", ".", ".", ".", "."]

    assert dep_a_a.maybe_get_value("a") == 3
    assert dep_a.maybe_get_value("a") == [".", ".", "."]

    dep_b.settle_all()

    assert dep_b.maybe_get_value("a") == [".", ".", ".", ".", "."]

    value_a.set_value("a", 4)

    assert dep_a.maybe_get_value("b") == [":", ":", ":"]
    assert dep_b.maybe_get_value("b") == [":", ":", ":"]

    assert dep_a_a.maybe_get_value("a") == 3  # time sensitive
    assert dep_a.maybe_get_value("a") == [".", ".", "."]  # time sensitive

    dep_a.settle_all()
    dep_b.settle_all()

    assert dep_a_a.maybe_get_value("a") == 3  # time sensitive
    assert dep_a.maybe_get_value("a") == [".", ".", ".", "."]
    assert dep_b.maybe_get_value("a") == [".", ".", ".", ".", "."]

    dep_a_a.settle_all()

    assert dep_a_a.maybe_get_value("a") == 4
