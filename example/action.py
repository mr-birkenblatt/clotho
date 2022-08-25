from typing import Any, Dict, Literal, Optional, TypedDict


MessageAction = TypedDict('MessageAction', {
    "text": str,
})
LinkAction = TypedDict('LinkAction', {
    "parent_ref": str,
    "user_ref": Optional[str],
    "user_name": Optional[str],
    "created_utc": float,
    "votes": Dict[str, int],
    "depth": int,
})
Action = TypedDict('Action', {
    "kind": Literal["message", "link"],
    "ref_id": str,
    "message": Optional[MessageAction],
    "link": Optional[LinkAction],
}, total=False)


def create_message_action(ref_id: str, text: str) -> Action:
    return {
        "kind": "message",
        "ref_id": ref_id,
        "message": {
            "text": text,
        },
    }


def create_link_action(
        ref_id: str,
        parent_ref: str,
        depth: int,
        user_ref: Optional[str],
        user_name: Optional[str],
        created_utc: float,
        votes: Dict[str, int]) -> Action:
    return {
        "kind": "link",
        "ref_id": ref_id,
        "link": {
            "parent_ref": parent_ref,
            "user_ref": user_ref,
            "user_name": user_name,
            "created_utc": created_utc,
            "votes": votes,
            "depth": depth,
        },
    }


def maybe_to_string(text: Any) -> Optional[str]:
    if text is None:
        return None
    return f"{text}"


def is_link_action(action: Action) -> bool:
    return action["kind"] == "link"


def is_message_action(action: Action) -> bool:
    return action["kind"] == "message"


def parse_action(obj: Dict[str, Any]) -> Action:
    kind = obj["kind"]
    if kind == "link":
        link = obj["link"]
        return create_link_action(
            f"{obj['ref_id']}",
            f"{link['parent_ref']}",
            int(link["depth"]),
            maybe_to_string(link.get("user_ref")),
            maybe_to_string(link.get("user_name")),
            float(link["created_utc"]),
            {
                f"{vote}": int(count)
                for (vote, count) in link["votes"].items()
            })
    if kind == "message":
        message = obj["message"]
        return create_message_action(f"{obj['ref_id']}", f"{message['text']}")
    raise ValueError(f"unkown kind: {kind}")
