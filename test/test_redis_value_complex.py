
import time
from typing import List, NamedTuple, Tuple

from effects.effects import EffectDependent
from effects.redis import ValueDependentRedisType, ValueRootRedisType


Link = NamedTuple('Link', [
    ("l_from", str),
    ("l_to", str),
])


def test_complex() -> None:
    links: ValueRootRedisType[Link, int] = ValueRootRedisType(
        "test", lambda key: f"link:{key.l_from}:{key.l_to}")

    def compute_destinations(
            obj: EffectDependent[Link, List[int]],
            parents: Tuple[ValueRootRedisType[Link, int]],
            key: Link) -> None:
        lks, = parents
        obj.set_value(key, sorted(lks.get_range(f"link:{key.l_from}:")))

    def compute_sources(
            obj: EffectDependent[Link, List[int]],
            parents: Tuple[ValueRootRedisType[Link, int]],
            key: Link) -> None:
        lks, = parents
        obj.set_value(key, sorted(lks.get_range("link:", f":{key.l_to}")))

    dests: ValueDependentRedisType[Link, List[int]] = ValueDependentRedisType(
        "test",
        lambda key: f"dests:{key.l_from}",
        (links,),
        compute_destinations,
        0.1)
    srcs: ValueDependentRedisType[Link, List[int]] = ValueDependentRedisType(
        "test",
        lambda key: f"srcs:{key.l_to}",
        (links,),
        compute_sources,
        0.1)

    assert dests.get_value(Link(l_from="a", l_to="b"), []) == []

    links.set_value(Link(l_from="a", l_to="b"), 1)

    assert srcs.get_value(Link(l_from="a", l_to="b"), []) == [1]

    links.set_value(Link(l_from="a", l_to="c"), 2)
    links.set_value(Link(l_from="a", l_to="d"), 3)
    links.set_value(Link(l_from="b", l_to="b"), 0)
    links.set_value(Link(l_from="c", l_to="b"), 1)
    links.set_value(Link(l_from="d", l_to="b"), 2)
    links.set_value(Link(l_from="b", l_to="a"), 5)
    links.set_value(Link(l_from="b", l_to="d"), 4)
    links.set_value(Link(l_from="a", l_to="b"), 6)
    time.sleep(0.2)

    assert dests.get_value(Link(l_from="a", l_to="b"), []) == [6, 2, 3]
    assert dests.get_value(Link(l_from="b", l_to="b"), []) == [0, 4, 5]
    assert dests.get_value(Link(l_from="c", l_to="b"), []) == [1]
    assert dests.get_value(Link(l_from="d", l_to="b"), []) == [2]
    assert srcs.get_value(Link(l_from="a", l_to="a"), []) == [5]
    assert srcs.get_value(Link(l_from="a", l_to="b"), []) == [0, 1, 2, 6]
    assert srcs.get_value(Link(l_from="a", l_to="c"), []) == [2]
    assert srcs.get_value(Link(l_from="a", l_to="d"), []) == [3, 4]
