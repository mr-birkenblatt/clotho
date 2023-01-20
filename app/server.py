# pylint: disable=unused-argument
import sys
import threading
from typing import TypedDict

import pandas as pd
from quick_server import create_server, QuickServer
from quick_server import QuickServerRequestHandler as QSRH
from quick_server import ReqArgs, WorkerArgs

from app.response_types import (
    LinkListResponse,
    LoginResponse,
    LogoutResponse,
    MessageResponse,
    TopicListResponse,
    TopicResponse,
)
from app.token import RedisTokenHandler
from misc.env import envload_int, envload_str
from misc.util import now_ts, to_list
from system.links.link import Link, LinkResponse, parse_vote_type, VT_UP
from system.links.scorer import get_scorer, Scorer
from system.links.store import get_link_store
from system.msgs.message import Message, MHash
from system.msgs.store import get_message_store
from system.namespace.store import get_namespace
from system.suggest.suggest import get_link_suggesters
from system.users.store import get_user_store
from system.users.user import MAX_USER_NAME_LEN, User


LinkQuery = TypedDict('LinkQuery', {
    "scorer": Scorer,
    "now": pd.Timestamp,
    "offset": int,
    "limit": int,
})


MAX_RESPONSE = 1024 * 100  # 100kB  # rough size
MAX_LINKS = 20


def setup(
        ns_name: str,
        addr: str,
        port: int,
        parallel: bool,
        deploy: bool) -> tuple[QuickServer, str, str]:
    server: QuickServer = create_server(
        (addr, port),
        parallel,
        thread_factory=threading.Thread,
        token_handler=RedisTokenHandler(),
        worker_constructor=None,
        soft_worker_death=True)

    prefix = "/api"

    server.suppress_noise = True

    def report_slow_requests(method_str: str, path: str) -> None:
        print(f"slow request {method_str} {path}")

    max_upload = 20 * 1024 * 1024 * 1024  # 20GiB
    server_timeout = 10 * 60
    server.report_slow_requests = report_slow_requests
    server.max_file_size = max_upload
    server.max_chunk_size = max_upload
    server.timeout = server_timeout
    server.socket.settimeout(server_timeout)

    if deploy:
        server.no_command_loop = True

    print(f"python version: {sys.version}")

    server.set_default_token_expiration(48 * 60 * 60)  # 2 days

    namespace = get_namespace(ns_name)
    message_store = get_message_store(namespace)
    link_store = get_link_store(namespace)
    user_store = get_user_store(namespace)
    link_suggesters = get_link_suggesters(namespace)

    def get_user(args: WorkerArgs) -> User:
        with server.get_token_obj(args["token"]) as obj:
            return user_store.get_user_by_id(
                user_store.get_id_from_name(obj["user"]))

    def get_maybe_user(args: WorkerArgs) -> User | None:
        if args.get("token") is None:
            return None
        return get_user(args)

    # *** user management ***

    @server.json_post(f"{prefix}/login")
    def _post_login(_req: QSRH, rargs: ReqArgs) -> LoginResponse:
        args = rargs["post"]
        user_name = args["user"]
        user_id = user_store.get_id_from_name(user_name)
        user = user_store.get_user_by_id(user_id)
        token = server.create_token()
        with server.get_token_obj(token) as obj:
            obj["user"] = user.get_name()
        return {
            "token": token,
            "user": user.get_name(),
            "userid": user.get_id(),
            "permissions": user.get_permissions(),
        }

    @server.json_post(f"{prefix}/logout")
    def _post_logout(_req: QSRH, rargs: ReqArgs) -> LogoutResponse:
        args = rargs["post"]
        token = args.get("token")
        if token is not None:
            with server.get_token_obj(token, 0) as _:
                pass
        return {
            "success": True,
        }

    @server.json_post(f"{prefix}/signup")
    def _post_signup(_req: QSRH, rargs: ReqArgs) -> LoginResponse:
        args = rargs["post"]
        user_name = args["user"]
        if len(user_name) > MAX_USER_NAME_LEN:
            raise ValueError(f"invalid user name: {user_name}")
        user_id = user_store.get_id_from_name(user_name)
        try:
            user_store.get_user_by_id(user_id)
            raise ValueError(f"user already exists: {user_name}")
        except KeyError:
            pass
        user = User(user_name, {
            "can_create_topic": False,
        })
        user_store.store_user(user)
        token = server.create_token()
        with server.get_token_obj(token) as obj:
            obj["user"] = user.get_name()
        return {
            "token": token,
            "user": user.get_name(),
            "userid": user.get_id(),
            "permissions": user.get_permissions(),
        }

    @server.json_post(f"{prefix}/user")
    def _post_user(_req: QSRH, rargs: ReqArgs) -> LoginResponse:
        args = rargs["post"]
        user = get_user(args)
        return {
            "token": args["token"],
            "user": user.get_name(),
            "userid": user.get_id(),
            "permissions": user.get_permissions(),
        }

    # *** interactions ***

    @server.json_post(f"{prefix}/topic")
    def _post_topic(_req: QSRH, rargs: ReqArgs) -> TopicResponse:
        args = rargs["post"]
        user = get_user(args)
        assert user.can_create_topic()
        topic = f"{args['topic']}"
        msg = Message(msg=topic)
        message_store.add_topic(msg)
        message_store.write_message(msg)
        return {
            "topic": topic,
            "hash": msg.get_hash().to_parseable(),
        }

    @server.json_post(f"{prefix}/message")
    def _post_message(_req: QSRH, rargs: ReqArgs) -> LinkResponse:
        args = rargs["post"]
        user = get_user(args)
        parent = MHash.parse(f"{args['parent']}")
        msg = Message(msg=args["msg"].strip().replace("\r", ""))
        if not msg.is_valid_message():
            raise ValueError("cannot create topic via /message use /topic")
        child = message_store.write_message(msg)
        link = link_store.get_link(parent, child)
        now = now_ts()
        link.add_vote(user_store, VT_UP, user, now)
        return link.get_response(user_store, who=user, now=now, meta="new")

    @server.json_post(f"{prefix}/vote")
    def _post_vote(_req: QSRH, rargs: ReqArgs) -> LinkResponse:
        args = rargs["post"]
        is_add = bool(args["isadd"])
        votes = to_list(args["votes"])
        user = get_user(args)
        parent = MHash.parse(f"{args['parent']}")
        child = MHash.parse(f"{args['child']}")
        link = link_store.get_link(parent, child)
        now = now_ts()
        for vtype in votes:
            vote_type = parse_vote_type(f"{vtype}")
            if is_add:
                link.add_vote(user_store, vote_type, user, now)
            else:
                link.remove_vote(user_store, vote_type, user, now)
        return link.get_response(user_store, who=user, now=now, meta="vote")

    # *** read only ***

    @server.json_get(f"{prefix}/topic")
    def _get_topic(_req: QSRH, rargs: ReqArgs) -> TopicListResponse:
        args = rargs["query"]
        offset = int(args["offset"])
        limit = min(int(args["limit"]), MAX_LINKS)
        return {
            "topics": {
                msg.get_hash().to_parseable(): msg.get_text()
                for msg in message_store.get_topics(offset, limit)
            },
        }

    @server.json_post(f"{prefix}/read")
    def _post_read(_req: QSRH, rargs: ReqArgs) -> MessageResponse:
        args = rargs["post"]
        hashes = to_list(args["hashes"])
        cur_length = 0
        msgs = {}
        skipped: list[str] = []
        for mhash in hashes:
            if cur_length > MAX_RESPONSE:
                skipped.append(MHash.parse(mhash).to_parseable())
                continue
            msg = message_store.read_message(MHash.parse(mhash))
            key = msg.get_hash().to_parseable()
            value = msg.get_text()
            msgs[key] = value
            cur_length += len(key) + len(value)
        return {
            "messages": msgs,
            "skipped": skipped,
        }

    def get_link_query_params(args: WorkerArgs) -> LinkQuery:
        scorer = get_scorer(args["scorer"])
        now = now_ts()
        offset = int(args["offset"])
        limit = min(int(args["limit"]), MAX_LINKS)
        return {
            "scorer": scorer,
            "now": now,
            "offset": offset,
            "limit": limit,
        }

    def enrich_links(
            other: MHash,
            link_query: LinkQuery,
            *,
            is_parent: bool,
            links: list[tuple[Link, str]],
            total: int) -> list[tuple[Link, str]]:
        limit = link_query["limit"]
        if len(links) >= limit:
            return links
        offset = link_query["offset"]
        cur_offset = offset + len(links)
        cur_limit = limit - len(links)
        if cur_limit == 0:
            return links
        six = 0
        cur_offset -= total
        suggestions: list[tuple[Link, str]] = []
        while cur_limit > 0 and six < len(link_suggesters):
            cur_suggester = link_suggesters[six]
            local_limit = cur_limit
            maxs = cur_suggester.max_suggestions()
            if maxs is not None:
                if cur_offset > maxs:
                    cur_offset -= maxs
                    six += 1
                    continue
                local_limit = min(local_limit, maxs - cur_offset)
            prev_size = len(suggestions)
            suggestions.extend((
                (link, cur_suggester.get_name())
                for link in cur_suggester.suggest_links(
                    other,
                    is_parent=is_parent,
                    offset=cur_offset,
                    limit=local_limit)
            ))
            if len(suggestions) <= prev_size:
                six += 1
                continue
            cur_offset += local_limit
            cur_limit -= local_limit
        return links + suggestions

    @server.json_post(f"{prefix}/children")
    def _post_children(_req: QSRH, rargs: ReqArgs) -> LinkListResponse:
        args = rargs["post"]
        muser = get_maybe_user(args)
        parent = MHash.parse(args["parent"])
        link_query = get_link_query_params(args)
        links = [
            (link, "child")
            for link in link_store.get_children(parent, **link_query)
        ]
        total = link_store.get_all_children_count(parent, link_query["now"])
        links = enrich_links(
            parent, link_query, is_parent=True, links=links, total=total)
        return {
            "links": [
                link.get_response(
                    user_store, who=muser, now=link_query["now"], meta=meta)
                for link, meta in links
            ],
            "next": link_query["offset"] + len(links),
        }

    @server.json_post(f"{prefix}/parents")
    def _post_parents(_req: QSRH, rargs: ReqArgs) -> LinkListResponse:
        args = rargs["post"]
        muser = get_maybe_user(args)
        child = MHash.parse(args["child"])
        link_query = get_link_query_params(args)
        links = [
            (link, "parent")
            for link in link_store.get_parents(child, **link_query)
        ]
        total = link_store.get_all_parents_count(child, link_query["now"])
        links = enrich_links(
            child, link_query, is_parent=False, links=links, total=total)
        return {
            "links": [
                link.get_response(
                    user_store, who=muser, now=link_query["now"], meta=meta)
                for link, meta in links
            ],
            "next": link_query["offset"] + len(links),
        }

    @server.json_post(f"{prefix}/userlinks")
    def _post_userlinks(_req: QSRH, rargs: ReqArgs) -> LinkListResponse:
        args = rargs["post"]
        muser = get_maybe_user(args)
        user = user_store.get_user_by_id(args["userid"])
        link_query = get_link_query_params(args)
        links = [
            (link, "user")
            for link in link_store.get_user_links(user, **link_query)
        ]
        return {
            "links": [
                link.get_response(
                    user_store, who=muser, now=link_query["now"], meta=meta)
                for link, meta in links
            ],
            "next": link_query["offset"] + len(links),
        }

    @server.json_post(f"{prefix}/link")
    def _post_link(_req: QSRH, rargs: ReqArgs) -> LinkResponse:
        args = rargs["post"]
        muser = get_maybe_user(args)
        parent = MHash.parse(args["parent"])
        child = MHash.parse(args["child"])
        now = now_ts()
        link = link_store.get_link(parent, child)
        return link.get_response(user_store, who=muser, now=now, meta="link")

    return server, prefix, ns_name


def setup_server(
        deploy: bool,
        ns_name: str | None,
        addr: str | None,
        port: int | None) -> tuple[QuickServer, str, str]:
    if ns_name is None:
        ns_name = envload_str("API_SERVER_NAMESPACE", default="default")
    if addr is None:
        addr = envload_str("API_SERVER_HOST", default="127.0.0.1")
    if port is None:
        port = envload_int("API_SERVER_PORT", default=8080)
    return setup(ns_name, addr, port, parallel=True, deploy=deploy)


def start(server: QuickServer, prefix: str, ns_name: str) -> None:
    addr, port = server.server_address
    print(
        f"starting API at http://{addr}:{port}{prefix}/ "
        f"for namespace {ns_name}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        print("shutting down..")
        server.server_close()
