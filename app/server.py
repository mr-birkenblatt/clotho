from typing import Optional
import sys
import threading
import pandas as pd
from quick_server import (
    WorkerArgs,
    create_server,
    get_time,
    get_worker_check,
    PreventDefaultResponse,
    QuickServer,
)
from quick_server import QuickServerRequestHandler as QSRH
from quick_server import ReqArgs, Response

from misc.env import envload_int, envload_str
from app.token import RedisTokenHandler
from app.response_types import (
    LoginResponse,
    TopicResponse,
)
from misc.util import to_list
from system.links.link import VT_UP, Link, LinkResponse, parse_vote_type
from system.links.store import get_default_link_store
from system.links.user import User
from system.msgs.message import MHash, Message
from system.msgs.store import get_default_message_store


def setup(addr: str, port: int, parallel: bool, deploy: bool) -> QuickServer:
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

    @server.json_post(f"{prefix}/login")
    def _post_login(_req: QSRH, rargs: ReqArgs) -> LoginResponse:
        args = rargs["post"]
        user = args["user"]
        token = server.create_token()
        with server.get_token_obj(token) as obj:
            obj["user"] = user
        return {
            "token": token,
            "user": user
        }

    def get_user(args: WorkerArgs) -> User:
        with server.get_token_obj(args["token"]) as obj:
            return User.parse_name(obj["user"])

    def now_ts() -> pd.Timestamp:
        return pd.Timestamp("now")

    message_store = get_default_message_store()
    link_store = get_default_link_store()

    @server.json_post(f"{prefix}/topic")
    def _post_topic(_req: QSRH, rargs: ReqArgs) -> TopicResponse:
        args = rargs["post"]
        user = get_user(args)
        assert user.can_create_topic()
        topic = f"{args['topic']}"
        thash = message_store.add_topic(Message(msg=topic))
        return {
            "topic": topic,
            "hash": thash.to_parseable(),
        }

    @server.json_post(f"{prefix}/message")
    def _post_message(_req: QSRH, rargs: ReqArgs) -> LinkResponse:
        args = rargs["post"]
        user = get_user(args)
        parent = MHash.parse(f"{args['parent']}")
        msg = Message(msg=args["msg"])
        child = message_store.write_message(msg)
        link = link_store.get_link(parent, child)
        now = now_ts()
        link.add_vote(VT_UP, user, now)
        return link.get_response(now)

    @server.json_post(f"{prefix}/vote")
    def _post_vote(_req: QSRH, rargs: ReqArgs) -> LinkResponse:
        args = rargs["post"]
        votes = to_list(args["votes"])
        user = get_user(args)
        parent = MHash.parse(f"{args['parent']}")
        child = MHash.parse(f"{args['child']}")
        link = link_store.get_link(parent, child)
        now = now_ts()
        for vtype in votes:
            link.add_vote(parse_vote_type(f"{vtype}"), user, now)
        return link.get_response(now)

    return server


def setup_server(
        deploy: bool,
        addr: Optional[str],
        port: Optional[int]) -> QuickServer:
    if addr is None:
        addr = envload_str("HOST", default="localhost")
    if port is None:
        port = envload_int("PORT", default=8080)
    return setup(addr, port, parallel=True, deploy=deploy)
