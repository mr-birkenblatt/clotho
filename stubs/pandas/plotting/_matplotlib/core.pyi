# Stubs for pandas.plotting._matplotlib.core (Python 3)
#
# NOTE: This dynamically typed stub was automatically generated by stubgen.
# pylint: disable=unused-argument,redefined-outer-name,invalid-name
# pylint: disable=relative-beyond-top-level,arguments-differ
# pylint: disable=no-member,too-few-public-methods,keyword-arg-before-vararg
# pylint: disable=super-init-not-called,abstract-method,redefined-builtin
# pylint: disable=unused-import,useless-import-alias,signature-differs
# pylint: disable=blacklisted-name,c-extension-no-member,import-error
from typing import Any, Optional

class MPLPlot:
    data: Any = ...
    by: Any = ...
    kind: Any = ...
    sort_columns: Any = ...
    subplots: Any = ...
    sharex: bool = ...
    sharey: Any = ...
    figsize: Any = ...
    layout: Any = ...
    xticks: Any = ...
    yticks: Any = ...
    xlim: Any = ...
    ylim: Any = ...
    title: Any = ...
    use_index: Any = ...
    fontsize: Any = ...
    rot: Any = ...
    grid: Any = ...
    legend: Any = ...
    legend_handles: Any = ...
    legend_labels: Any = ...
    ax: Any = ...
    fig: Any = ...
    axes: Any = ...
    errors: Any = ...
    secondary_y: Any = ...
    colormap: Any = ...
    table: Any = ...
    kwds: Any = ...

    def __init__(
            self, data: Any, kind: Optional[Any] = ...,
            by: Optional[Any] = ..., subplots: bool = ...,
            sharex: Optional[Any] = ..., sharey: bool = ...,
            use_index: bool = ..., figsize: Optional[Any] = ...,
            grid: Optional[Any] = ..., legend: bool = ...,
            rot: Optional[Any] = ..., ax: Optional[Any] = ...,
            fig: Optional[Any] = ..., title: Optional[Any] = ...,
            xlim: Optional[Any] = ..., ylim: Optional[Any] = ...,
            xticks: Optional[Any] = ..., yticks: Optional[Any] = ...,
            sort_columns: bool = ..., fontsize: Optional[Any] = ...,
            secondary_y: bool = ..., colormap: Optional[Any] = ...,
            table: bool = ..., layout: Optional[Any] = ...,
            **kwds: Any) -> None:
        ...

    @property
    def nseries(self) -> Any:
        ...

    def draw(self) -> None:
        ...

    def generate(self) -> None:
        ...

    @property
    def result(self) -> Any:
        ...

    @property
    def legend_title(self) -> Any:
        ...

    def plt(self) -> Any:
        ...

    @classmethod
    def get_default_ax(cls, ax: Any) -> None:
        ...

    def on_right(self, i: Any) -> Any:
        ...


class PlanePlot(MPLPlot):
    x: Any = ...
    y: Any = ...

    def __init__(self, data: Any, x: Any, y: Any, **kwargs: Any) -> None:
        ...

    @property
    def nseries(self) -> Any:
        ...


class ScatterPlot(PlanePlot):
    c: Any = ...

    def __init__(
            self, data: Any, x: Any,
            y: Any, s: Optional[Any] = ...,
            c: Optional[Any] = ..., **kwargs: Any) -> None:
        ...


class HexBinPlot(PlanePlot):
    C: Any = ...

    def __init__(
            self, data: Any, x: Any,
            y: Any, C: Optional[Any] = ..., **kwargs: Any) -> None:
        ...


class LinePlot(MPLPlot):
    data: Any = ...
    x_compat: Any = ...

    def __init__(self, data: Any, **kwargs: Any) -> None:
        ...


class AreaPlot(LinePlot):
    def __init__(self, data: Any, **kwargs: Any) -> None:
        ...


class BarPlot(MPLPlot):
    bar_width: Any = ...
    tick_pos: Any = ...
    bottom: Any = ...
    left: Any = ...
    log: Any = ...
    tickoffset: Any = ...
    lim_offset: Any = ...
    ax_pos: Any = ...

    def __init__(self, data: Any, **kwargs: Any) -> None:
        ...


class BarhPlot(BarPlot):
    ...


class PiePlot(MPLPlot):
    def __init__(
            self, data: Any, kind: Optional[Any] = ..., **kwargs: Any) -> None:
        ...
