import argparse
import os
import subprocess
import sys
from typing import Callable, TYPE_CHECKING

import numpy as np
import torch

from misc.util import json_compact, json_read, python_module


if TYPE_CHECKING:
    from model.embedding import ProviderRole
    from system.namespace.namespace import Namespace


def run_index_lookup(
        namespace: 'Namespace',
        role: 'ProviderRole',
        shards: list[int],
        embed: torch.Tensor | None,
        count: int,
        precise: bool,
        process_out: Callable[[list[int], str], None],
        on_err: Callable[[BaseException], None]) -> None:
    try:
        module = python_module()
        python_exec = sys.executable
        cmd = [python_exec, "-m", module]
        cmd.extend(["--namespace", namespace.get_name()])
        cmd.extend(["--role", role])
        shards_str = [f"{shard}" for shard in shards]
        cmd.extend(["--shards", f"{','.join(shards_str)}"])
        if embed is not None:
            cmd.extend(["--count", f"{count}"])
            if precise:
                cmd.append("--precise")
        res = subprocess.run(
            cmd,
            capture_output=True,
            check=False,
            input=None if embed is None else serialize_embedding(embed),
            encoding="utf-8")
        if res.returncode != 0:
            raise ValueError(
                f"Error in command ({cmd})\n"
                f"STDOUT_START\n{res.stdout}\nSTDOUT_END\n"
                f"STDERR_START\n{res.stderr}\nSTDERR_END")
        if res.stderr:
            print(f"STDERR_START({cmd})\n{res.stderr}\nSTDERR_STOP")
        for line in res.stdout.splitlines(keepends=False):
            process_out(shards, line)
    except BaseException as e:  # pylint: disable=broad-except
        on_err(e)


def parse_args() -> argparse.Namespace:
    description = (
        "Build index or retrieve neighbors. "
        "Expects the embedding as JSON in stdin if --count is set.")
    parser = argparse.ArgumentParser(
        prog=f"python -m {python_module()}",
        description=description)
    parser.add_argument("--namespace", help="the namespace")
    parser.add_argument("--role", help="the provider role")
    parser.add_argument(
        "--shards",
        help="',' separated list of shards to build or retrieve from")
    parser.add_argument(
        "--count", default=None, type=int, help="how many neighbors to return")
    parser.add_argument(
        "--precise",
        default=False,
        action="store_true",
        help="whether to ignore the index during lookup")
    return parser.parse_args()


def serialize_embedding(embed: torch.Tensor) -> str:
    arr = embed.double().detach().numpy().astype(np.float64).tolist()
    return json_compact(arr).decode("utf-8")


def deserialize_embedding(content: str) -> torch.Tensor:
    return torch.DoubleTensor(json_read(content.encode("utf-8")))


def run() -> None:
    from model.embedding import get_provider_role
    from system.embedding.index_lookup import CachedIndexEmbeddingStore
    from system.embedding.store import get_embed_store
    from system.namespace.store import get_namespace

    args = parse_args()
    ns_name: str = args.namespace
    namespace = get_namespace(ns_name)
    role = get_provider_role(args.role)
    shards = [int(shard) for shard in f"{args.shards}".split(",")]
    embed_store = get_embed_store(namespace)
    if not isinstance(embed_store, CachedIndexEmbeddingStore):
        raise ValueError(
            f"invalid embed store {embed_store.__class__.__name__} "
            f"in namespace {ns_name}")
    index_lookup: CachedIndexEmbeddingStore = embed_store
    if args.count is None:
        for shard in shards:
            if index_lookup.can_build_index(role, shard):
                index_lookup.set_index_lock_state(role, shard, os.getpid())
                index_lookup.proc_build_index_shard(role, shard)
                # NOTE: only remove the lock if we were is successful
                index_lookup.set_index_lock_state(role, shard, None)
    else:
        count = args.count
        embed = deserialize_embedding(sys.stdin.read())
        ignore_index = args.precise
        for shard in shards:
            for mhash, distance in index_lookup.proc_get_closest(
                    role, shard, embed, count, ignore_index=ignore_index):
                print(f"{mhash.to_parseable()},{distance}")


if __name__ == "__main__":
    run()
