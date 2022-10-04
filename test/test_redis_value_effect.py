from effects.redis import SetRootRedisType, ValueRootRedisType


def test_value() -> None:
    root: ValueRootRedisType[str, int] = ValueRootRedisType(
        "test", lambda key: key)
    assert root.maybe_get_value("a") is None
    root.set_value("a", 5)
    assert root.maybe_get_value("a") == 5
    root.set_value("b", 10)
    assert root.maybe_get_value("b") == 10
    root.set_value("a", 13)
    root.set_value("a", 3)
    assert root.maybe_get_value("a") == 3
    assert root.maybe_get_value("b") == 10


def test_set() -> None:
    root: SetRootRedisType[str] = SetRootRedisType("test", lambda key: key)
    assert root.maybe_get_value("foo") == set()
    assert not root.add_value("foo", "a")
    assert root.maybe_get_value("foo") == {"a"}
    assert not root.add_value("bar", "b")
    assert root.maybe_get_value("bar") == {"b"}
    assert not root.add_value("foo", "c")
    assert root.maybe_get_value("foo") == {"a", "c"}
    assert root.add_value("foo", "c")
    assert root.maybe_get_value("foo") == {"a", "c"}
    assert root.maybe_get_value("bar") == {"b"}
    assert root.remove_value("foo", "c")
    assert root.maybe_get_value("foo") == {"a"}
    assert not root.remove_value("foo", "b")
    assert root.maybe_get_value("foo") == {"a"}
    assert root.remove_value("foo", "a")
    assert root.maybe_get_value("foo") == set()
    assert root.maybe_get_value("bar") == {"b"}
