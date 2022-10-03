from effects.redis import SetRootRedisType, StrRootRedisType


def test_value() -> None:
    root: StrRootRedisType[str, int] = StrRootRedisType(
        "test", lambda key: key)
    assert root.get_value("a") is None
    root.set_value("a", 5)
    assert root.get_value("a") == 5
    root.set_value("b", 10)
    assert root.get_value("b") == 10
    root.set_value("a", 3)
    assert root.get_value("a") == 3
    assert root.get_value("b") == 10


def test_set() -> None:
    root: SetRootRedisType[str] = SetRootRedisType("test", lambda key: key)
    assert root.get_value("foo") == set()
    root.add_value("foo", "a")
    assert root.get_value("foo") == {"a"}
    root.add_value("bar", "b")
    assert root.get_value("bar") == {"b"}
    root.add_value("foo", "c")
    assert root.get_value("foo") == {"a", "c"}
    assert root.get_value("bar") == {"b"}
    root.remove_value("foo", "c")
    assert root.get_value("foo") == {"a"}
    root.remove_value("foo", "b")
    assert root.get_value("foo") == {"a"}
    root.remove_value("foo", "a")
    assert root.get_value("foo") == set()
    assert root.get_value("bar") == {"b"}
