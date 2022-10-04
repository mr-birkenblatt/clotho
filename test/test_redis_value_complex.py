
import time
from typing import List, NamedTuple, Tuple

from effects.effects import EffectDependent
from effects.redis import (
    ListDependentRedisType,
    ValueDependentRedisType,
    ValueRootRedisType,
)


Link = NamedTuple('Link', [
    ("l_from", str),
    ("l_to", str),
])
FLink = NamedTuple('FLink', [
    ("l_from", str),
])
TLink = NamedTuple('TLink', [
    ("l_to", str),
])


def test_complex() -> None:
    links: ValueRootRedisType[Link, int] = ValueRootRedisType(
        "test", lambda key: f"link:{key.l_from}:{key.l_to}")

    def compute_destinations(
            obj: EffectDependent[FLink, List[int], Link],
            parents: Tuple[ValueRootRedisType[Link, int]],
            key: Link) -> None:
        lks, = parents
        fkey = FLink(l_from=key.l_from)
        obj.set_value(fkey, sorted(lks.get_range(f"link:{key.l_from}:")))

    def compute_sources(
            obj: EffectDependent[TLink, List[int], Link],
            parents: Tuple[ValueRootRedisType[Link, int]],
            key: Link) -> None:
        lks, = parents
        tkey = TLink(l_to=key.l_to)
        obj.set_value(tkey, sorted(lks.get_range("link:", f":{key.l_to}")))

    dests: ValueDependentRedisType[FLink, List[int], Link] = \
        ValueDependentRedisType(
            "test",
            lambda key: f"dests:{key.l_from}",
            (links,),
            compute_destinations,
            0.1)
    srcs: ValueDependentRedisType[TLink, List[int], Link] = \
        ValueDependentRedisType(
            "test",
            lambda key: f"srcs:{key.l_to}",
            (links,),
            compute_sources,
            0.1)

    assert dests.get_value(FLink(l_from="a"), []) == []

    links.set_value(Link(l_from="a", l_to="b"), 1)

    assert srcs.get_value(TLink(l_to="b"), []) == []  # time sensitive

    links.set_value(Link(l_from="a", l_to="c"), 2)
    links.set_value(Link(l_from="a", l_to="d"), 3)
    links.set_value(Link(l_from="b", l_to="b"), 0)
    links.set_value(Link(l_from="c", l_to="b"), 1)
    links.set_value(Link(l_from="d", l_to="b"), 2)
    links.set_value(Link(l_from="b", l_to="a"), 5)
    links.set_value(Link(l_from="b", l_to="d"), 4)
    links.set_value(Link(l_from="a", l_to="b"), 6)
    time.sleep(0.2)

    assert dests.get_value(FLink(l_from="a"), []) == [2, 3, 6]
    assert dests.get_value(FLink(l_from="b"), []) == [0, 4, 5]
    assert dests.get_value(FLink(l_from="c"), []) == [1]
    assert dests.get_value(FLink(l_from="d"), []) == [2]
    assert srcs.get_value(TLink(l_to="a"), []) == [5]
    assert srcs.get_value(TLink(l_to="b"), []) == [0, 1, 2, 6]
    assert srcs.get_value(TLink(l_to="c"), []) == [2]
    assert srcs.get_value(TLink(l_to="d"), []) == [3, 4]


def test_complex_list() -> None:
    links: ValueRootRedisType[Link, int] = ValueRootRedisType(
        "test", lambda key: f"link:{key.l_from}:{key.l_to}")

    def compute_destinations(
            obj: EffectDependent[FLink, List[str], Link],
            parents: Tuple[ValueRootRedisType[Link, int]],
            key: Link) -> None:
        lks, = parents
        fkey = FLink(l_from=key.l_from)
        obj.set_value(fkey, sorted((
            f"{val}" for val in lks.get_range(f"link:{key.l_from}:"))))

    def compute_sources(
            obj: EffectDependent[TLink, List[str], Link],
            parents: Tuple[ValueRootRedisType[Link, int]],
            key: Link) -> None:
        lks, = parents
        tkey = TLink(l_to=key.l_to)
        obj.set_value(tkey, sorted((
            f"{val}" for val in lks.get_range("link:", f":{key.l_to}"))))

    dests: ListDependentRedisType[FLink, Link] = \
        ListDependentRedisType(
            "test",
            lambda key: f"dests:{key.l_from}",
            (links,),
            compute_destinations,
            0.1)
    srcs: ListDependentRedisType[TLink, Link] = \
        ListDependentRedisType(
            "test",
            lambda key: f"srcs:{key.l_to}",
            (links,),
            compute_sources,
            0.1)

    assert dests.get_value(FLink(l_from="a"), []) == []

    links.set_value(Link(l_from="a", l_to="b"), 1)

    assert srcs.get_value(TLink(l_to="b"), []) == []  # time sensitive

    links.set_value(Link(l_from="a", l_to="c"), 2)
    links.set_value(Link(l_from="a", l_to="d"), 3)
    links.set_value(Link(l_from="b", l_to="b"), 0)
    links.set_value(Link(l_from="c", l_to="b"), 1)
    links.set_value(Link(l_from="d", l_to="b"), 2)
    links.set_value(Link(l_from="b", l_to="a"), 5)
    links.set_value(Link(l_from="b", l_to="d"), 4)
    links.set_value(Link(l_from="a", l_to="b"), 6)
    time.sleep(0.2)

    assert dests.get_value(FLink(l_from="a"), []) == ["2", "3", "6"]
    assert dests.get_value(FLink(l_from="b"), []) == ["0", "4", "5"]
    assert dests.get_value(FLink(l_from="c"), []) == ["1"]
    assert dests.get_value(FLink(l_from="d"), []) == ["2"]
    assert srcs.get_value(TLink(l_to="a"), []) == ["5"]
    assert srcs.get_value(TLink(l_to="b"), []) == ["0", "1", "2", "6"]
    assert srcs.get_value(TLink(l_to="c"), []) == ["2"]
    assert srcs.get_value(TLink(l_to="d"), []) == ["3", "4"]

    arr = ["0", "1", "2", "6"]
    assert srcs.get_value_range(TLink(l_to="b"), 0, 4) == arr
    assert srcs.get_value_range(TLink(l_to="b"), None, 0) == []
    assert srcs.get_value_range(TLink(l_to="b"), None, 1) == ["0"]
    assert srcs.get_value_range(TLink(l_to="b"), None, None) == arr
    assert srcs.get_value_range(TLink(l_to="b"), 0, None) == arr
    assert srcs.get_value_range(TLink(l_to="b"), 0, -1) == ["0", "1", "2"]
    assert srcs.get_value_range(TLink(l_to="b"), 0, 0) == []
    assert srcs.get_value_range(TLink(l_to="b"), 0, 1) == ["0"]
    assert srcs.get_value_range(TLink(l_to="b"), 0, 2) == ["0", "1"]
    assert srcs.get_value_range(TLink(l_to="b"), 1, 1) == []
    assert srcs.get_value_range(TLink(l_to="b"), 1, 2) == ["1"]
    assert srcs.get_value_range(TLink(l_to="b"), 1, 3) == ["1", "2"]
    assert srcs.get_value_range(TLink(l_to="b"), 2, 4) == ["2", "6"]
    assert srcs.get_value_range(TLink(l_to="b"), 2, None) == ["2", "6"]
    assert srcs.get_value_range(TLink(l_to="b"), -3, None) == arr[-3:]
    assert srcs.get_value_range(TLink(l_to="b"), 1, -1) == arr[1:-1]
    assert srcs.get_value_range(TLink(l_to="b"), -3, -1) == arr[-3:-1]
    assert srcs.get_value_range(TLink(l_to="b"), -3, 3) == arr[-3:3]
    assert srcs.get_value_range(TLink(l_to="b"), 3, 5) == ["6"]
    assert srcs.get_value_range(TLink(l_to="b"), 4, 5) == []

    assert srcs[TLink(l_to="b"), 0:2] == ["0", "1"]
    assert srcs[TLink(l_to="b"), None:2] == ["0", "1"]
    assert srcs[TLink(l_to="b"), 1:3] == ["1", "2"]
    assert srcs[TLink(l_to="b"), 1:] == ["1", "2", "6"]
    assert srcs[TLink(l_to="b"), 3:] == ["6"]
    assert len(srcs[TLink(l_to="b"), 4:]) == 0
    assert srcs[TLink(l_to="b"), ::-1] == arr[::-1]
    assert srcs[TLink(l_to="b"), 5:0:-1] == arr[5:0:-1]
    assert srcs[TLink(l_to="b"), 4:0:-1] == arr[4:0:-1]
    assert srcs[TLink(l_to="b"), 3:0:-1] == arr[3:0:-1]
    assert srcs[TLink(l_to="b"), 2:0:-1] == arr[2:0:-1]
    assert srcs[TLink(l_to="b"), 1:0:-1] == arr[1:0:-1]
    assert srcs[TLink(l_to="b"), 3:1:-1] == arr[3:1:-1]
    assert srcs[TLink(l_to="b"), 4:2:-1] == arr[4:2:-1]
    assert srcs[TLink(l_to="b"), 0:5:-1] == arr[0:5:-1]
    assert srcs[TLink(l_to="b"), 0:4:-1] == arr[0:4:-1]
    assert srcs[TLink(l_to="b"), 0:3:-1] == arr[0:3:-1]
    assert srcs[TLink(l_to="b"), 0:2:-1] == arr[0:2:-1]
    assert srcs[TLink(l_to="b"), 0:1:-1] == arr[0:1:-1]
    assert srcs[TLink(l_to="b"), 0:0:-1] == arr[0:0:-1]
    assert srcs[TLink(l_to="b"), :2:-1] == arr[:2:-1]
    assert srcs[TLink(l_to="b"), 2::-1] == arr[2::-1]
    assert srcs[TLink(l_to="b"), -1:2:-1] == arr[-1:2:-1]
    assert srcs[TLink(l_to="b"), -2:1:-1] == arr[-2:1:-1]
    assert srcs[TLink(l_to="b"), 1:4:2] == ["1", "6"]
