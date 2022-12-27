# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from _typeshed import Incomplete


def dump_tensorboard_summary(graph_executor, logdir) -> None: ...


def visualize(
    graph, name_prefix: str = ..., pb_graph: Incomplete | None = ...,
    executors_it: Incomplete | None = ...): ...


def visualize_graph_executor(state, name_prefix, pb_graph, inline_graph): ...


def visualize_rec(
    graph, value_map, name_prefix, pb_graph,
    executors_it: Incomplete | None = ...): ...
