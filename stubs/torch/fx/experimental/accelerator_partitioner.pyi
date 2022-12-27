# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


import torch
from _typeshed import Incomplete
from torch.fx.experimental.partitioner_utils import Device as Device


        NodeLatency as NodeLatency, Partition as Partition,
        PartitionMode as PartitionMode,
        PartitionerConfig as PartitionerConfig,
        get_extra_size_of as get_extra_size_of,
        get_latency_of_partitioned_graph as get_latency_of_partitioned_graph,
        get_partition_to_latency_mapping as get_partition_to_latency_mapping
from typing import Dict, List, NamedTuple, Tuple

from torch.fx.graph_module import GraphModule as GraphModule
from torch.fx.node import map_arg as map_arg
from torch.fx.node import Node as Node
from torch.fx.passes.graph_manipulation import (
    get_size_of_all_nodes as get_size_of_all_nodes,
)
from torch.fx.passes.split_module import split_module as split_module


class DAGNode:
    submodule_node: Incomplete
    input_nodes: Incomplete
    output_nodes: Incomplete
    logical_device_ids: Incomplete
    size_bytes: Incomplete

    def __init__(
        self, submodule_node: Node, input_nodes: List[Node],
        output_nodes: List[Node], logical_device_ids: List[int],
        size_bytes: int) -> None: ...


class DAG:
    nodes: Incomplete
    def __init__(self) -> None: ...

    def create_node(
        self, submodule_node: Node, input_nodes: List[Node],
        output_nodes: List[Node], logical_devices: List[int],
        size_bytes: int) -> None: ...


class PartitionResult(NamedTuple):
    dag: DAG
    module_with_submodules: GraphModule


def reset_partition_device(partitions) -> None: ...


def combine_two_partitions(
    partition_0: Partition, partition_1: Partition,
    partitions: List[Partition]) -> None: ...


def set_parents_and_children(partitions: List[Partition]) -> None: ...


def reorganize_partitions(partitions: List[Partition]) -> None: ...


def get_bfs_level_partition(partitions: List[Partition]) -> None: ...


def get_node_to_partition_mapping(
    partitions: List[Partition]) -> Dict[Node, int]: ...


def get_logical_id_to_device(devices: List[Device]) -> Dict[int, Device]: ...


def get_device_partition_stats(
    partitions: List[Partition], devices: List[Device]) -> Tuple[Dict[Device,
        List[Partition]], Dict[Device, int], List[Partition]]: ...


def get_device_to_partitions_mapping(
    partitions: List[Partition], devices: List[Device]): ...


def check_dependency(partition): ...


class Partitioner:
    partitions: Incomplete
    node_to_partition: Incomplete
    devices: Incomplete
    def __init__(self) -> None: ...
    graph_module: Incomplete
    torch_module: Incomplete

    def partition_graph(
        self, fx_module: GraphModule, torch_module: torch.nn.Module,
        partitioner_config: PartitionerConfig) -> PartitionResult: ...

    def find_single_partition(
        self, total_size_of_graph, logical_device_id: int = ...) -> None: ...

    def size_based_partition(self) -> None: ...
    def saturate_host(self) -> None: ...
    def do_partition(self) -> GraphModule: ...
    def dump_dag(self, module_with_submodules: GraphModule) -> DAG: ...
    def create_partition(self) -> Partition: ...
    def create_single_node_partition(self, node) -> None: ...
    def sparse_nn_partition(self, available_mem_bytes: int) -> None: ...

    def cost_aware_partition(
        self, transfer_rate_bytes_per_sec: float,
        node_to_latency_mapping: Dict[Node, NodeLatency]) -> None: ...

    def kl_based_partition(
        self, transfer_rate_bytes_per_sec: float,
        node_to_latency_mapping: Dict[Node, NodeLatency]) -> None: ...

    def aot_based_partition(
        self, node_to_partition_mapping,
        partition_to_logical_device_mapping) -> None: ...
