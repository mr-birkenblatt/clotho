
import time
from typing import NamedTuple

import pandas as pd

from effects.effects import set_old_threshold
from effects.redis import (
    ListDependentRedisType,
    ValueDependentRedisType,
    ValueRootRedisType,
)
from misc.redis import REDIS_TEST_CONFIG
from misc.util import from_timestamp, json_compact, json_read


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
    now = 1670580000.0
    set_old_threshold(0.1)
    links: ValueRootRedisType[Link, int] = ValueRootRedisType(
        "links",
        REDIS_TEST_CONFIG,
        "test",
        lambda key: f"link:{key.l_from}:{key.l_to}")

    def compute_destinations(key: FLink, now: pd.Timestamp | None) -> None:
        dests.set_value(
            key, sorted(links.get_range(f"link:{key.l_from}:")), now)

    def compute_sources(key: TLink, now: pd.Timestamp | None) -> None:
        srcs.set_value(
            key, sorted(links.get_range("link:", f":{key.l_to}")), now)

    dests: ValueDependentRedisType[FLink, list[int]] = \
        ValueDependentRedisType(
            "dests",
            REDIS_TEST_CONFIG,
            "test",
            lambda key: f"dests:{key.l_from}",
            lambda key: json_compact(key.l_from),
            lambda obj: FLink(json_read(obj)),
            "",
            "obs",
            "pen",
            parents=(links,),
            convert=lambda pkey: FLink(l_from=pkey.l_from),
            effect=compute_destinations)
    srcs: ValueDependentRedisType[TLink, list[int]] = \
        ValueDependentRedisType(
            "srcs",
            REDIS_TEST_CONFIG,
            "test",
            lambda key: f"srcs:{key.l_to}",
            lambda key: json_compact(key.l_to),
            lambda obj: TLink(json_read(obj)),
            "",
            "obs",
            "pen",
            parents=(links,),
            convert=lambda pkey: TLink(l_to=pkey.l_to),
            effect=compute_sources)

    now_ts = from_timestamp(now)
    assert dests.get_value(FLink(l_from="a"), [], now_ts) == []

    links.set_value(Link(l_from="a", l_to="b"), 1, now_ts)

    assert srcs.get_value(TLink(l_to="b"), [], now_ts) == [1]

    links.set_value(Link(l_from="a", l_to="c"), 2, now_ts)
    links.set_value(Link(l_from="a", l_to="d"), 3, now_ts)
    links.set_value(Link(l_from="b", l_to="b"), 0, now_ts)
    links.set_value(Link(l_from="c", l_to="b"), 1, now_ts)
    links.set_value(Link(l_from="d", l_to="b"), 2, now_ts)
    links.set_value(Link(l_from="b", l_to="a"), 5, now_ts)
    links.set_value(Link(l_from="b", l_to="d"), 4, now_ts)
    links.set_value(Link(l_from="a", l_to="b"), 6, now_ts)

    time.sleep(0.2)
    now += 0.2
    now_ts = from_timestamp(now)

    assert dests.get_value(FLink(l_from="a"), [], now_ts) == [2, 3, 6]
    assert dests.get_value(FLink(l_from="b"), [], now_ts) == [0, 4, 5]
    assert dests.get_value(FLink(l_from="c"), [], now_ts) == [1]
    assert dests.get_value(FLink(l_from="d"), [], now_ts) == [2]
    assert srcs.get_value(TLink(l_to="a"), [], now_ts) == [5]
    assert srcs.get_value(TLink(l_to="b"), [], now_ts) == [0, 1, 2, 6]
    assert srcs.get_value(TLink(l_to="c"), [], now_ts) == [2]
    assert srcs.get_value(TLink(l_to="d"), [], now_ts) == [3, 4]


def test_complex_list() -> None:
    now = 1670580000.0
    set_old_threshold(0.1)
    links: ValueRootRedisType[Link, int] = ValueRootRedisType(
        "links",
        REDIS_TEST_CONFIG,
        "test",
        lambda key: f"link:{key.l_from}:{key.l_to}")

    def compute_destinations(key: FLink, now: pd.Timestamp | None) -> None:
        dests.set_value(key, sorted((
            f"{val}" for val in links.get_range(f"link:{key.l_from}:"))), now)

    def compute_sources(key: TLink, now: pd.Timestamp | None) -> None:
        srcs.set_value(
            key,
            sorted((
                f"{val}" for val in links.get_range("link:", f":{key.l_to}"))),
            now)

    dests: ListDependentRedisType[FLink] = \
        ListDependentRedisType(
            "dests",
            REDIS_TEST_CONFIG,
            "test",
            lambda key: f"dests:{key.l_from}",
            lambda key: json_compact(key.l_from),
            lambda obj: FLink(json_read(obj)),
            "",
            "obs",
            "pen",
            parents=(links,),
            convert=lambda pkey: FLink(l_from=pkey.l_from),
            effect=compute_destinations,
            empty=b"")
    srcs: ListDependentRedisType[TLink] = \
        ListDependentRedisType(
            "srcs",
            REDIS_TEST_CONFIG,
            "test",
            lambda key: f"srcs:{key.l_to}",
            lambda key: json_compact(key.l_to),
            lambda obj: TLink(json_read(obj)),
            "",
            "obs",
            "pen",
            parents=(links,),
            convert=lambda pkey: TLink(l_to=pkey.l_to),
            effect=compute_sources,
            empty=b"")

    now_ts = from_timestamp(now)

    assert dests.get_value(FLink(l_from="a"), [], now_ts) == []

    links.set_value(Link(l_from="a", l_to="b"), 1, now_ts)

    assert srcs.get_value(TLink(l_to="b"), [], now_ts) == ["1"]

    links.set_value(Link(l_from="a", l_to="c"), 2, now_ts)
    links.set_value(Link(l_from="a", l_to="d"), 3, now_ts)
    links.set_value(Link(l_from="b", l_to="b"), 0, now_ts)
    links.set_value(Link(l_from="c", l_to="b"), 1, now_ts)
    links.set_value(Link(l_from="d", l_to="b"), 2, now_ts)
    links.set_value(Link(l_from="b", l_to="a"), 5, now_ts)
    links.set_value(Link(l_from="b", l_to="d"), 4, now_ts)
    links.set_value(Link(l_from="a", l_to="b"), 6, now_ts)

    time.sleep(0.2)
    now += 0.2
    now_ts = from_timestamp(now)

    assert dests.get_value(FLink(l_from="a"), [], now_ts) == ["2", "3", "6"]
    assert dests.get_value(FLink(l_from="b"), [], now_ts) == ["0", "4", "5"]
    assert dests.get_value(FLink(l_from="c"), [], now_ts) == ["1"]
    assert dests.get_value(FLink(l_from="d"), [], now_ts) == ["2"]
    assert srcs.get_value(TLink(l_to="a"), [], now_ts) == ["5"]
    assert srcs.get_value(TLink(l_to="b"), [], now_ts) == ["0", "1", "2", "6"]
    assert srcs.get_value(TLink(l_to="c"), [], now_ts) == ["2"]
    assert srcs.get_value(TLink(l_to="d"), [], now_ts) == ["3", "4"]

    arr = ["0", "1", "2", "6"]
    assert srcs.get_value_range(TLink(l_to="b"), 0, 4, now_ts) == arr
    assert srcs.get_value_range(TLink(l_to="b"), None, 0, now_ts) == []
    assert srcs.get_value_range(TLink(l_to="b"), None, 1, now_ts) == ["0"]
    assert srcs.get_value_range(TLink(l_to="b"), None, None, now_ts) == arr
    assert srcs.get_value_range(TLink(l_to="b"), 0, None, now_ts) == arr
    assert srcs.get_value_range(
        TLink(l_to="b"), 0, -1, now_ts) == ["0", "1", "2"]
    assert srcs.get_value_range(TLink(l_to="b"), 0, 0, now_ts) == []
    assert srcs.get_value_range(TLink(l_to="b"), 0, 1, now_ts) == ["0"]
    assert srcs.get_value_range(TLink(l_to="b"), 0, 2, now_ts) == ["0", "1"]
    assert srcs.get_value_range(TLink(l_to="b"), 1, 1, now_ts) == []
    assert srcs.get_value_range(TLink(l_to="b"), 1, 2, now_ts) == ["1"]
    assert srcs.get_value_range(TLink(l_to="b"), 1, 3, now_ts) == ["1", "2"]
    assert srcs.get_value_range(TLink(l_to="b"), 2, 4, now_ts) == ["2", "6"]
    assert srcs.get_value_range(TLink(l_to="b"), 2, None, now_ts) == ["2", "6"]
    assert srcs.get_value_range(TLink(l_to="b"), -3, None, now_ts) == arr[-3:]
    assert srcs.get_value_range(TLink(l_to="b"), 1, -1, now_ts) == arr[1:-1]
    assert srcs.get_value_range(TLink(l_to="b"), -3, -1, now_ts) == arr[-3:-1]
    assert srcs.get_value_range(TLink(l_to="b"), -3, 3, now_ts) == arr[-3:3]
    assert srcs.get_value_range(TLink(l_to="b"), 3, 5, now_ts) == ["6"]
    assert srcs.get_value_range(TLink(l_to="b"), 4, 5, now_ts) == []

    assert srcs[TLink(l_to="b"), 0:2, now_ts] == ["0", "1"]
    assert len(srcs[TLink(l_to="b"), 2:0, now_ts]) == 0
    assert srcs[TLink(l_to="b"), None:2, now_ts] == ["0", "1"]
    assert srcs[TLink(l_to="b"), 1:3, now_ts] == ["1", "2"]
    assert srcs[TLink(l_to="b"), 1:, now_ts] == ["1", "2", "6"]
    assert srcs[TLink(l_to="b"), 3:, now_ts] == ["6"]
    assert len(srcs[TLink(l_to="b"), 4:, now_ts]) == 0
    assert srcs[TLink(l_to="b"), ::-1, now_ts] == arr[::-1]
    assert srcs[TLink(l_to="b"), 5:0:-1, now_ts] == arr[5:0:-1]
    assert srcs[TLink(l_to="b"), 4:0:-1, now_ts] == arr[4:0:-1]
    assert srcs[TLink(l_to="b"), 3:0:-1, now_ts] == arr[3:0:-1]
    assert srcs[TLink(l_to="b"), 2:0:-1, now_ts] == arr[2:0:-1]
    assert srcs[TLink(l_to="b"), 1:0:-1, now_ts] == arr[1:0:-1]
    assert srcs[TLink(l_to="b"), 3:1:-1, now_ts] == arr[3:1:-1]
    assert srcs[TLink(l_to="b"), 4:2:-1, now_ts] == arr[4:2:-1]
    assert srcs[TLink(l_to="b"), 0:5:-1, now_ts] == arr[0:5:-1]
    assert srcs[TLink(l_to="b"), 0:4:-1, now_ts] == arr[0:4:-1]
    assert srcs[TLink(l_to="b"), 0:3:-1, now_ts] == arr[0:3:-1]
    assert srcs[TLink(l_to="b"), 0:2:-1, now_ts] == arr[0:2:-1]
    assert srcs[TLink(l_to="b"), 0:1:-1, now_ts] == arr[0:1:-1]
    assert srcs[TLink(l_to="b"), 0:0:-1, now_ts] == arr[0:0:-1]
    assert srcs[TLink(l_to="b"), 3:-1:-1, now_ts] == arr[3:-1:-1]
    assert srcs[TLink(l_to="b"), 3:-2:-1, now_ts] == arr[3:-2:-1]
    assert srcs[TLink(l_to="b"), 3:-3:-1, now_ts] == arr[3:-3:-1]
    assert srcs[TLink(l_to="b"), 3:-4:-1, now_ts] == arr[3:-4:-1]
    assert srcs[TLink(l_to="b"), :2:-1, now_ts] == arr[:2:-1]
    assert srcs[TLink(l_to="b"), 2::-1, now_ts] == arr[2::-1]
    assert srcs[TLink(l_to="b"), -1:2:-1, now_ts] == arr[-1:2:-1]
    assert srcs[TLink(l_to="b"), -2:1:-1, now_ts] == arr[-2:1:-1]
    assert srcs[TLink(l_to="b"), 1:4:2, now_ts] == ["1", "6"]

    assert len(srcs[TLink(l_to="d"), 2:0, now_ts]) == 0
