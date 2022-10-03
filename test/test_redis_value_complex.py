
import time
from typing import List, Tuple, TypedDict

from effects.effects import EffectDependent
from effects.redis import ValueDependentRedisType, ValueRootRedisType


Link = TypedDict('Link', {
    "from": str,
    "to": str,
})


def test_complex() -> None:
    links: ValueRootRedisType[Link, int] = ValueRootRedisType(
        "test", lambda key: f"link:{key['from']}:{key['to']}")

    def compute_destinations(
            obj: EffectDependent[Link, List[int]],
            parents: Tuple[ValueRootRedisType[Link, int]],
            key: Link) -> None:
        lks, = parents
        obj.set_value(key, sorted(lks.get_range(f"link:{key['from']}:")))

    def compute_sources(
            obj: EffectDependent[Link, List[int]],
            parents: Tuple[ValueRootRedisType[Link, int]],
            key: Link) -> None:
        lks, = parents
        obj.set_value(key, sorted(lks.get_range("link:", f":{key['to']}")))

    dests: ValueDependentRedisType[Link, List[int]] = ValueDependentRedisType(
        "test",
        lambda key: f"dests:{key['from']}",
        (links,),
        compute_destinations,
        0.1)
    srcs: ValueDependentRedisType[Link, List[int]] = ValueDependentRedisType(
        "test",
        lambda key: f"srcs:{key['to']}",
        (links,),
        compute_sources,
        0.1)

    assert dests.get_value({"from": "a", "to": "b"}, []) == []

    links.set_value({"from": "a", "to": "b"}, 1)

    assert srcs.get_value({"from": "a", "to": "b"}, []) == [1]

    links.set_value({"from": "a", "to": "c"}, 2)
    links.set_value({"from": "a", "to": "d"}, 3)
    links.set_value({"from": "b", "to": "b"}, 0)
    links.set_value({"from": "c", "to": "b"}, 1)
    links.set_value({"from": "d", "to": "b"}, 2)
    links.set_value({"from": "b", "to": "a"}, 5)
    links.set_value({"from": "b", "to": "d"}, 4)
    links.set_value({"from": "a", "to": "b"}, 6)
    time.sleep(0.2)

    assert dests.get_value({"from": "a", "to": "b"}, []) == [6, 2, 3]
    assert dests.get_value({"from": "b", "to": "b"}, []) == [0, 4, 5]
    assert dests.get_value({"from": "c", "to": "b"}, []) == [1]
    assert dests.get_value({"from": "d", "to": "b"}, []) == [2]
    assert srcs.get_value({"from": "a", "to": "a"}, []) == [5]
    assert srcs.get_value({"from": "a", "to": "b"}, []) == [0, 1, 2, 6]
    assert srcs.get_value({"from": "a", "to": "c"}, []) == [2]
    assert srcs.get_value({"from": "a", "to": "d"}, []) == [3, 4]
