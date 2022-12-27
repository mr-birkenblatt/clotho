# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from enum import Enum
from typing import Dict, List, NamedTuple, Set

from _typeshed import Incomplete
from torch.fx.node import map_arg as map_arg
from torch.fx.node import Node as Node


class Partition:
    nodes: Incomplete
    partition_id: Incomplete
    parents: Incomplete
    children: Incomplete
    bfs_level: int
    used_mem_bytes: int
    logical_device_ids: Incomplete
    def __init__(self, partition_id: int) -> None: ...
    def recalculate_mem_size(self) -> None: ...
    def add_node(self, node): ...
    def remove_node(self, node): ...


class Device(NamedTuple):
    name: str
    available_mem_bytes: int
    logical_id: int


class NodeLatency(NamedTuple):
    mem_latency_sec: float
    computer_latency_sec: float


class PartitionLatency(NamedTuple):
    mem_latency_sec: float
    computer_latency_sec: float
    overall_latency_sec: float


class PartitionMode(Enum):
    size_based: int
    sparse_nn: int
    cost_aware: int
    kl_based: int
    aot_based: int


class PartitionerConfig(NamedTuple):
    devices: List[Device]
    mode: PartitionMode
    transfer_rate_bytes_per_sec: float
    node_to_latency_mapping: Dict[Node, NodeLatency]
    node_to_partition_mapping: Dict[Node, int]
    partition_to_logical_device_mapping: Dict[int, List[int]]
    saturate_host: bool


def get_extra_size_of(node: Node, nodes: Set[Node]) -> int: ...


def get_latency_of_one_partition(
    partition: Partition, node_to_latency_mapping: Dict[Node,
            NodeLatency]) -> PartitionLatency: ...


def get_partition_to_latency_mapping(
    partitions: List[Partition], node_to_latency_mapping: Dict[Node,
            NodeLatency]) -> Dict[Partition, PartitionLatency]: ...


def get_comm_latency_between(
    parent_partition: Partition, child_partition: Partition,
    transfer_rate_bytes_per_sec: float): ...


def get_latency_of_partitioned_graph(
    partitions: List[Partition],
    partition_to_latency_mapping: Dict[Partition, PartitionLatency],
    transfer_rate_bytes_per_sec: float): ...
