import time

import pandas as pd

from example.loader import process_action_file
from system.links.store import get_link_store
from system.msgs.store import get_message_store
from system.users.store import get_user_store


def test_loader() -> None:
    message_store = get_message_store("ram")
    link_store = get_link_store("redis")
    user_store = get_user_store("ram")
    now = pd.Timestamp("2022-08-22", tz="UTC")
    reference_time = time.monotonic()
    process_action_file(
        "test/data/loader.jsonl",
        message_store=message_store,
        link_store=link_store,
        user_store=user_store,
        now=now,
        reference_time=reference_time,
        roots={"news"})

    # assert False
