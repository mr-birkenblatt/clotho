from typing import Iterable

from system.msgs.message import Message, MHash
from system.msgs.store import MessageStore, RandomGeneratingFunction


class RamMessageStore(MessageStore):
    def __init__(self) -> None:
        super().__init__()
        self._msgs: dict[MHash, Message] = {}
        self._topics: list[Message] = []

    def write_message(self, message: Message) -> MHash:
        mhash = message.get_hash()
        self._msgs[mhash] = message
        return mhash

    def read_message(self, message_hash: MHash) -> Message:
        return self._msgs[message_hash]

    def add_topic(self, topic: Message) -> MHash:
        self._topics.append(topic)
        return topic.get_hash()

    def get_topics(
            self,
            offset: int,
            limit: int | None) -> list[Message]:
        if limit is None:
            return self._topics[offset:]
        return self._topics[offset:offset + limit]

    def get_topics_count(self) -> int:
        return len(self._topics)

    def do_get_random_messages(
            self,
            get_random: RandomGeneratingFunction,
            count: int) -> Iterable[MHash]:
        keys = list(self._msgs.keys())
        yield from (
            keys[get_random(high=len(keys), for_row=cur_row)]
            for cur_row in range(count))

    def enumerate_messages(self, *, progress_bar: bool) -> Iterable[MHash]:
        if not progress_bar:
            yield from list(self._msgs.keys())
            return
        # FIXME: add stubs
        from tqdm.auto import tqdm  # type: ignore

        with tqdm(total=len(self._msgs)) as pbar:
            for mhash in list(self._msgs.keys()):
                yield mhash
                pbar.update(1)

    def get_message_count(self) -> int:
        return len(self._msgs)
