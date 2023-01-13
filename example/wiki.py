import bz2
import gzip
from typing import Iterable, Set, TypedDict
from xml.etree import ElementTree as ET

import pandas as pd

from example.action import Action, create_link_action, create_message_action
from misc.util import json_compact, now_ts, to_timestamp
from system.links.link import VT_UP


REF_WIKI = "_wikipedia"
TOPIC_WIKI = "t/wikipedia"


WikiState = TypedDict('WikiState', {
    "text": str | None,
    "title": str | None,
    "url": str | None,
    "user": str | None,
    "page_type": int,
    "is_invalid": bool,
    "timestamp": pd.Timestamp | None,
})


def init_wiki_state() -> WikiState:
    return {
        "text": None,
        "title": None,
        "url": None,
        "user": None,
        "page_type": -1,
        "is_invalid": False,
        "timestamp": None,
    }


def finalize_action(state: WikiState) -> Iterable[Action]:
    if state["is_invalid"]:
        yield from []
        return
    if state["title"] is None or state["text"] is None:
        print(f"WARNING: page has no content? {state}")
        yield from []
        return
    title = state["title"]
    text = state["text"]
    if state["url"] is None:
        text = f"{text} (from {title})"
    else:
        text = f"{text} (from {title}: {state['url']})"
    yield create_message_action(title, text)
    timestamp = to_timestamp(
        now_ts() if state["timestamp"] is None else state["timestamp"])
    yield create_link_action(
        title,
        REF_WIKI,
        depth=0,
        user_ref=state["user"],
        user_name=state["user"],
        created_utc=timestamp,
        votes={VT_UP: 10})


def strip_tag(tag: str) -> str:
    rix = tag.rfind("}")
    if rix >= 0:
        tag = tag[rix + 1:]
    return tag


IGNORED_ABSTRACT_TAGS: Set[str] = {"link", "sublink", "anchor", "links"}


def process_abstract_event(elem: ET.Element, state: WikiState) -> bool:
    tag = strip_tag(str(elem.tag))
    if tag == "abstract":
        text = elem.text
        if text is not None:
            state["text"] = text
            if len(text) < 20:
                state["is_invalid"] = True
        return False
    if tag == "title":
        state["title"] = elem.text
        return False
    if tag == "url":
        state["url"] = elem.text
        return False
    if tag == "doc":
        return True
    if tag not in IGNORED_ABSTRACT_TAGS:
        print(elem.tag, repr(elem.text), json_compact(elem.attrib))
    return False


IGNORED_FULL_TAGS: Set[str] = set()


def process_full_event(elem: ET.Element, state: WikiState) -> bool:
    tag = strip_tag(elem.tag)
    if tag == "title":
        state["title"] = elem.text
        return False
    if tag == "text":
        text = elem.text
        if text is not None:
            state["text"] = text[:20]
            if text.startswith("#REDIRECT"):
                state["is_invalid"] = True
        return False
    if tag == "redirect":
        state["is_invalid"] = True
        return False
    if tag == "username":
        state["user"] = elem.text
        return False
    if tag == "timestamp":
        text = elem.text
        if text is not None:
            state["timestamp"] = pd.to_datetime(text)
        return False
    if tag == "ns":
        text = elem.text
        if text is not None:
            state["page_type"] = int(text)
        return False
    if tag == "page":
        return True
    if tag not in IGNORED_FULL_TAGS:
        print(elem.tag, repr(elem.text), json_compact(elem.attrib))
    return False


def read_wiki(fname: str, *, is_abstract: bool) -> Iterable[Action]:
    process_event = \
        process_abstract_event if is_abstract else process_full_event
    compress = gzip if is_abstract else bz2
    state = init_wiki_state()
    yield create_message_action(REF_WIKI, TOPIC_WIKI)
    with compress.open(fname, mode="rt", encoding="utf-8") as gin:
        for _, elem in ET.iterparse(gin, events=("end",)):
            if process_event(elem, state):
                yield from finalize_action(state)
                state = init_wiki_state()
            elem.clear()
