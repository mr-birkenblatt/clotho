import time

import pandas as pd

from effects.effects import set_old_threshold
from effects.redis import (
    ListDependentRedisType,
    ValueDependentRedisType,
    ValueRootRedisType,
)
from misc.util import from_timestamp, json_compact, json_read


def test_dependent() -> None:
    now = 1670580000.0
    set_old_threshold(0.1)
    value_a: ValueRootRedisType[str, int] = ValueRootRedisType(
        "test", lambda key: f"count:{key}")
    value_b: ValueRootRedisType[str, str] = ValueRootRedisType(
        "test", lambda key: f"name:{key}")

    def key_id(key: str) -> str:
        return key

    def update_a(key: str, now_ts: pd.Timestamp | None) -> None:
        dep_a.set_value(
            key,
            [value_b.get_value(key, "MISSING")] * value_a.get_value(key, 0),
            now_ts)

    def update_b(key: str, now_ts: pd.Timestamp | None) -> None:
        ref = dep_b.maybe_get_value(key, now_ts)
        old = dep_b.update_value(
            key, len(value_b.get_value(key, "MISSING")), now_ts)
        assert (old is None and ref is None) or old == ref

    dep_a: ValueDependentRedisType[str, list[str]] = \
        ValueDependentRedisType(
            "test",
            lambda key: f"list:{key}",
            json_compact,
            json_read,
            "",
            "obs",
            "pen",
            parents=(value_a, value_b),
            convert=key_id,
            effect=update_a)
    dep_b: ValueDependentRedisType[str, int] = \
        ValueDependentRedisType(
            "test",
            lambda key: f"len:{key}",
            json_compact,
            json_read,
            "",
            "obs",
            "pen",
            parents=(value_b,),
            convert=key_id,
            effect=update_b)

    assert value_a.maybe_get_value("a") is None
    value_a.set_value("a", 2, from_timestamp(now))
    assert value_a.maybe_get_value("a") == 2
    value_b.on_update("a", from_timestamp(now))
    assert dep_a.maybe_get_value("a", from_timestamp(now)) is None
    assert dep_b.maybe_get_value("a", from_timestamp(now)) is None
    time.sleep(0.2)
    now += 0.2
    assert dep_a.maybe_get_value("a", from_timestamp(now)) \
        == ["MISSING", "MISSING"]
    assert dep_b.maybe_get_value("a", from_timestamp(now)) == 7

    value_a.set_value("a", 3, from_timestamp(now))
    value_b.set_value("a", "abc", from_timestamp(now))
    value_a.set_value("b", 2, from_timestamp(now))
    value_b.set_value("b", "defg", from_timestamp(now))
    assert dep_a.maybe_get_value("b", from_timestamp(now)) is None
    assert dep_b.maybe_get_value("b", from_timestamp(now)) is None
    time.sleep(0.2)
    now += 0.2
    assert dep_a.maybe_get_value("a", from_timestamp(now)) \
        == ["abc", "abc", "abc"]
    assert dep_b.maybe_get_value("a", from_timestamp(now)) == 3
    assert dep_a.maybe_get_value("b", from_timestamp(now)) == ["defg", "defg"]
    assert dep_b.maybe_get_value("b", from_timestamp(now)) == 4

    def update_a_a(key: str, now_ts: pd.Timestamp | None) -> None:
        dep_a_a.set_value(
            key, "-".join(dep_a.get_value(key, [], now_ts)), now_ts)

    def update_a_b(key: str, now_ts: pd.Timestamp | None) -> None:
        new = "-".join(dep_a.get_value(key, [], now_ts))
        old = dep_a_b.maybe_get_value(key, now_ts)
        if dep_a_b.set_new_value(key, new, now_ts):
            assert dep_a_b.maybe_get_value(key, now_ts) == new
        else:
            assert dep_a_b.maybe_get_value(key, now_ts) == old

    def update_a_c(key: str, now_ts: pd.Timestamp | None) -> None:
        val = dep_a.get_value(key, [], now_ts)
        if val:
            dep_a_c.set_value(key, "-".join(val), now_ts)

    dep_a_a: ValueDependentRedisType[str, str] = ValueDependentRedisType(
        "test",
        lambda key: f"concat:{key}",
        json_compact,
        json_read,
        "",
        "obs",
        "pen",
        parents=(dep_a,),
        convert=key_id,
        effect=update_a_a)
    dep_a_b: ValueDependentRedisType[str, str] = ValueDependentRedisType(
        "test",
        lambda key: f"first:{key}",
        json_compact,
        json_read,
        "",
        "obs",
        "pen",
        parents=(dep_a,),
        convert=key_id,
        effect=update_a_b)
    dep_a_c: ValueDependentRedisType[str, str] = ValueDependentRedisType(
        "test",
        lambda key: f"never:{key}",
        json_compact,
        json_read,
        "",
        "obs",
        "pen",
        parents=(dep_a,),
        convert=key_id,
        effect=update_a_c)

    dep_a.on_update("a", from_timestamp(now))
    dep_a.on_update("b", from_timestamp(now))
    dep_a.on_update("c", from_timestamp(now))
    time.sleep(0.2)
    now += 0.2

    assert dep_a_a.maybe_get_value("a", from_timestamp(now)) == "abc-abc-abc"
    assert dep_a_b.maybe_get_value("a", from_timestamp(now)) == "abc-abc-abc"
    assert dep_a_c.maybe_get_value("a", from_timestamp(now)) == "abc-abc-abc"

    value_a.set_value("a", 7, from_timestamp(now))
    value_b.set_value("a", "=", from_timestamp(now))
    value_a.set_value("a", 5, from_timestamp(now))
    value_b.set_value("a", "+", from_timestamp(now))
    assert dep_a_a.maybe_get_value("a", from_timestamp(now)) \
        == "abc-abc-abc"  # time sensitive
    time.sleep(0.4)
    now += 0.4
    assert dep_a_a.maybe_get_value("a", from_timestamp(now)) == "+-+-+-+-+"
    assert dep_a_b.maybe_get_value("a", from_timestamp(now)) == "abc-abc-abc"
    assert dep_a.maybe_get_value("a", from_timestamp(now)) \
        == ["+", "+", "+", "+", "+"]
    assert dep_b.maybe_get_value("a", from_timestamp(now)) == 1
    assert value_a.maybe_get_value("a") == 5
    assert value_b.maybe_get_value("a") == "+"
    assert value_a.maybe_get_value("b") == 2
    assert value_b.maybe_get_value("b") == "defg"
    assert dep_a_a.maybe_get_value("b", from_timestamp(now)) == "defg-defg"
    assert dep_a.maybe_get_value("b", from_timestamp(now)) == ["defg", "defg"]
    assert dep_b.maybe_get_value("b", from_timestamp(now)) == 4

    assert dep_a_a.get_value("c", "MISSING", from_timestamp(now)) == ""
    assert dep_a_b.maybe_get_value("c", from_timestamp(now)) == ""
    assert dep_a_c.get_value("c", "MISSING", from_timestamp(now)) == "MISSING"


def test_dependent_list() -> None:
    now = 1670580000.0
    set_old_threshold(0.1)
    value_a: ValueRootRedisType[str, int] = ValueRootRedisType(
        "test", lambda key: f"count:{key}")
    value_b: ValueRootRedisType[str, str] = ValueRootRedisType(
        "test", lambda key: f"name:{key}")

    def key_id(key: str) -> str:
        return key

    def update_a(key: str, now_ts: pd.Timestamp | None) -> None:
        ref = dep_a.maybe_get_value(key, now_ts)
        old = dep_a.update_value(
            key,
            [value_b.get_value(key, "MISSING")] * value_a.get_value(key, 0),
            now_ts)
        assert (old == [] and ref is None) or old == ref

    def update_b(key: str, now_ts: pd.Timestamp | None) -> None:
        arr = [value_b.get_value(key, "MISSING")] * value_a.get_value(key, 0)
        old = dep_b.maybe_get_value(key, now_ts)
        print(f"x {key} {old} {arr}")
        if dep_b.set_new_value(key, arr, now_ts):
            assert dep_b.maybe_get_value(key, now_ts) == arr
        else:
            assert dep_b.maybe_get_value(key, now_ts) == old

    dep_a = ListDependentRedisType(
        "test",
        lambda key: f"slista:{key}",
        json_compact,
        json_read,
        "val",
        "obs",
        "pen",
        parents=(value_a, value_b),
        convert=key_id,
        effect=update_a)
    dep_b = ListDependentRedisType(
        "test",
        lambda key: f"slistb:{key}",
        json_compact,
        json_read,
        "val",
        "obs",
        "pen",
        parents=(value_a, value_b),
        convert=key_id,
        effect=update_b)

    def update_a_a(key: str, now_ts: pd.Timestamp | None) -> None:
        dep_a_a.set_value(key, len(dep_a.get_value(key, [], now_ts)), now_ts)

    dep_a_a: ValueDependentRedisType[str, int] = ValueDependentRedisType(
        "test",
        lambda key: f"counta:{key}",
        json_compact,
        json_read,
        "val",
        "obs",
        "pen",
        parents=(dep_a,),
        convert=key_id,
        effect=update_a_a)

    value_a.set_value("a", 5, from_timestamp(now))
    value_b.set_value("a", ".", from_timestamp(now))
    value_a.set_value("b", 3, from_timestamp(now))
    value_b.set_value("b", ":", from_timestamp(now))

    assert dep_a_a.maybe_get_value(
        "b", from_timestamp(now)) is None  # time sensitive
    assert dep_a.maybe_get_value(
        "b", from_timestamp(now)) is None  # time sensitive
    assert dep_b.maybe_get_value(
        "b", from_timestamp(now)) is None  # time sensitive
    now += 0.1
    assert dep_a_a.maybe_get_value(
        "a", from_timestamp(now)) is None  # time sensitive
    assert dep_b.maybe_get_value("a", None) \
        == [".", ".", ".", ".", "."]

    value_a.set_value("a", 3, from_timestamp(now))

    assert dep_b.maybe_get_value("a", from_timestamp(now)) \
        == [".", ".", ".", ".", "."]

    now += 0.1

    assert dep_b.maybe_get_value("a", from_timestamp(now)) \
        == [".", ".", ".", ".", "."]

    assert dep_a_a.maybe_get_value("a", None) == 3
    assert dep_a.maybe_get_value("a", None) == [".", ".", "."]

    now += 0.1

    assert dep_b.maybe_get_value("a", None) == [".", ".", ".", ".", "."]

    value_a.set_value("a", 4, from_timestamp(now))

    assert dep_a.maybe_get_value("b", from_timestamp(now)) == [":", ":", ":"]
    assert dep_b.maybe_get_value("b", from_timestamp(now)) == [":", ":", ":"]

    assert dep_a_a.maybe_get_value("a", from_timestamp(now)) \
        == 3  # time sensitive
    assert dep_a.maybe_get_value("a", from_timestamp(now)) \
        == [".", ".", "."]  # time sensitive

    now += 0.1

    assert dep_a_a.maybe_get_value("a", from_timestamp(now)) \
        == 3  # time sensitive
    assert dep_a.maybe_get_value("a", None) == [".", ".", ".", "."]
    assert dep_b.maybe_get_value("a", None) == [".", ".", ".", ".", "."]

    now += 0.1

    assert dep_a_a.maybe_get_value("a", None) == 4

    value_a.set_value("a", 6, from_timestamp(now + 1.0))
    assert dep_a.maybe_get_value("a", from_timestamp(now)) \
        == [".", ".", ".", "."]

    now += 0.1

    assert dep_a.maybe_get_value("a", from_timestamp(now)) \
        == [".", ".", ".", "."]

    time.sleep(0.1)

    assert dep_a.maybe_get_value("a", from_timestamp(now)) \
        == [".", ".", ".", ".", ".", "."]
