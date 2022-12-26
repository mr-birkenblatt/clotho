from enum import Enum


class ObservationType(Enum):
    OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT: int
    OUTPUT_SHARE_OBSERVER_WITH_INPUT: int
