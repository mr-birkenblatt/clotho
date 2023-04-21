from effects.redis import SetRootRedisType, ValueRootRedisType
from misc.redis import REDIS_TEST_CONFIG
from misc.util import from_timestamp


def test_value() -> None:
    root: ValueRootRedisType[str, int] = ValueRootRedisType(
        "root", REDIS_TEST_CONFIG, "test", lambda key: key)
    now_ts = from_timestamp(1670580000.0)
    assert root.maybe_get_value("a") is None
    root.set_value("a", 5, now_ts)
    assert root.maybe_get_value("a") == 5
    assert root.update_value("b", 10, now_ts) is None
    assert root.maybe_get_value("b") == 10
    root.set_value("a", 13, now_ts)
    root.set_value("a", 3, now_ts)
    assert root.maybe_get_value("a") == 3
    assert root.maybe_get_value("b") == 10
    root.set_value("a", 4, now_ts)
    assert root.update_value("a", 5, now_ts) == 4
    assert root.maybe_get_value("c") is None
    assert root.set_new_value("c", 10, now_ts)
    assert root.maybe_get_value("c") == 10
    assert not root.set_new_value("c", 15, now_ts)
    assert root.maybe_get_value("c") == 10


def test_set() -> None:
    now_ts = from_timestamp(1670580000.0)
    root: SetRootRedisType[str] = SetRootRedisType(
        "root", REDIS_TEST_CONFIG, "test", lambda key: key)
    assert root.maybe_get_value("foo") == set()
    assert not root.add_value("foo", "a", now_ts)
    assert root.maybe_get_value("foo") == {"a"}
    assert not root.add_value("bar", "b", now_ts)
    assert root.maybe_get_value("bar") == {"b"}
    assert not root.add_value("foo", "c", now_ts)
    assert root.maybe_get_value("foo") == {"a", "c"}
    assert root.add_value("foo", "c", now_ts)
    assert root.maybe_get_value("foo") == {"a", "c"}
    assert root.maybe_get_value("bar") == {"b"}
    assert root.remove_value("foo", "c", now_ts)
    assert root.maybe_get_value("foo") == {"a"}
    assert not root.remove_value("foo", "b", now_ts)
    assert root.maybe_get_value("foo") == {"a"}
    assert root.remove_value("foo", "a", now_ts)
    assert root.maybe_get_value("foo") == set()
    assert root.maybe_get_value("bar") == {"b"}
