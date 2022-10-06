from typing import Dict, List, TypedDict

from system.links.link import LinkResponse
from system.users.user import Permissions


LoginResponse = TypedDict('LoginResponse', {
    "token": str,
    "user": str,
    "permissions": Permissions,
})
TopicResponse = TypedDict('TopicResponse', {
    "topic": str,
    "hash": str,
})
TopicListResponse = TypedDict('TopicListResponse', {
    "topics": Dict[str, str],
})
MessageResponse = TypedDict('MessageResponse', {
    "messages": Dict[str, str],
    "skipped": List[str],
})
LinkListResponse = TypedDict('LinkListResponse', {
    "links": List[LinkResponse],
    "next": int,
})
