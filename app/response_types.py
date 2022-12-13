from typing import TypedDict

from system.links.link import LinkResponse
from system.users.user import Permissions


LoginResponse = TypedDict('LoginResponse', {
    "token": str,
    "user": str,
    "userid": str,
    "permissions": Permissions,
})
LogoutResponse = TypedDict('LogoutResponse', {
    "success": bool,
})
TopicResponse = TypedDict('TopicResponse', {
    "topic": str,
    "hash": str,
})
TopicListResponse = TypedDict('TopicListResponse', {
    "topics": dict[str, str],
})
MessageResponse = TypedDict('MessageResponse', {
    "messages": dict[str, str],
    "skipped": list[str],
})
LinkListResponse = TypedDict('LinkListResponse', {
    "links": list[LinkResponse],
    "next": int,
})
