from effects.redis import StrRootRedisType


def test_value() -> None:
    root: StrRootRedisType[str, int] = StrRootRedisType(
        "test", lambda key: key)
    assert root.get_value("a") is None
    root.set_value("a", 5)
    assert root.get_value("a") == 5
