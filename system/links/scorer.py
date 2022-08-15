from typing import cast, Dict, get_args, Literal

import pandas as pd

from misc.util import to_timestamp
from system.links.link import Link, VT_ACK, VT_DOWN, VT_SKIP, VT_UP

ScorerName = Literal[
    "new",
    "top",
    "best",
]
VALID_SCORER_NAMES = set(get_args(ScorerName))


class Scorer:  # pylint: disable=too-few-public-methods
    def get_score(self, link: Link, now: pd.Timestamp) -> float:
        raise NotImplementedError()


class NewScorer(Scorer):  # pylint: disable=too-few-public-methods
    def get_score(self, link: Link, now: pd.Timestamp) -> float:
        return to_timestamp(link.get_votes(VT_UP).get_first_vote_time(now))


class TopScorer(Scorer):  # pylint: disable=too-few-public-methods
    def get_score(self, link: Link, now: pd.Timestamp) -> float:
        return link.get_votes(VT_UP).get_total_votes()


class BestScorer(Scorer):  # pylint: disable=too-few-public-methods
    def get_score(self, link: Link, now: pd.Timestamp) -> float:
        ups = link.get_votes(VT_UP)
        downs = link.get_votes(VT_DOWN)
        acks = link.get_votes(VT_ACK)
        skips = link.get_votes(VT_SKIP)
        v_up = ups.get_total_votes() + ups.get_adjusted_daily_votes(now)
        v_down = downs.get_total_votes() + downs.get_adjusted_daily_votes(now)
        v_ack = acks.get_total_votes() + acks.get_adjusted_daily_votes(now)
        v_skip = skips.get_total_votes() + skips.get_adjusted_daily_votes(now)
        return max(v_up + 0.5 * v_ack, 0.5 * (v_down + 0.5 * v_skip))


SCORERS: Dict[ScorerName, Scorer] = {
    "new": NewScorer(),
    "top": TopScorer(),
    "best": BestScorer(),
}


def get_scorer(name: str) -> Scorer:
    assert name in VALID_SCORER_NAMES
    return SCORERS[cast(ScorerName, name)]
