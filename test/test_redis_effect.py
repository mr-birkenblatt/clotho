from effects.redis import StrRootRedisType


def test_value() -> None:
    root: StrRootRedisType[str, int] = StrRootRedisType(
        "test", lambda key: key)
    root.set_value("a", 5)
    root.get_value("a")
