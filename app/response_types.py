from typing import Dict, TypedDict


LoginResponse = TypedDict('LoginResponse', {
    "token": str,
    "user": str,
})
TopicResponse = TypedDict('TopicResponse', {
    "topic": str,
    "hash": str,
})
