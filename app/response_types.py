from typing import Dict, TypedDict

from system.links.link import VoteDict, VoteType


LoginResponse = TypedDict('LoginResponse', {
    "token": str,
    "user": str,
})
TopicResponse = TypedDict('TopicResponse', {
    "topic": str,
    "hash": str,
})
