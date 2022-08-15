from typing import Dict, List, TypedDict

from system.links.link import LinkResponse

LoginResponse = TypedDict('LoginResponse', {
    "token": str,
    "user": str,
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
