# Stubs for pandas.plotting._misc (Python 3)
#
# NOTE: This dynamically typed stub was automatically generated by stubgen.
# pylint: disable=unused-argument,redefined-outer-name,invalid-name
# pylint: disable=relative-beyond-top-level,arguments-differ
# pylint: disable=super-init-not-called

from typing import Any, Optional


def table(
        ax: Any, data: Any, rowLabels: Optional[Any] = ...,
        colLabels: Optional[Any] = ..., **kwargs: Any) -> Any:
    ...


def register(explicit: bool = ...) -> None:
    ...


def deregister() -> None:
    ...


def scatter_matrix(
        frame: Any, alpha: float = ..., figsize: Optional[Any] = ...,
        ax: Optional[Any] = ..., grid: bool = ..., diagonal: str = ...,
        marker: str = ..., density_kwds: Optional[Any] = ...,
        hist_kwds: Optional[Any] = ..., range_padding: float = ...,
        **kwds: Any) -> Any:
    ...


def radviz(
        frame: Any, class_column: Any, ax: Optional[Any] = ...,
        color: Optional[Any] = ..., colormap: Optional[Any] = ...,
        **kwds: Any) -> Any:
    ...


def andrews_curves(
        frame: Any, class_column: Any, ax: Optional[Any] = ...,
        samples: int = ..., color: Optional[Any] = ...,
        colormap: Optional[Any] = ..., **kwds: Any) -> Any:
    ...


def bootstrap_plot(
        series: Any, fig: Optional[Any] = ..., size: int = ...,
        samples: int = ..., **kwds: Any) -> Any:
    ...


def parallel_coordinates(
        frame: Any, class_column: Any, cols: Optional[Any] = ...,
        ax: Optional[Any] = ..., color: Optional[Any] = ...,
        use_columns: bool = ..., xticks: Optional[Any] = ...,
        colormap: Optional[Any] = ..., axvlines: bool = ...,
        axvlines_kwds: Optional[Any] = ..., sort_labels: bool = ...,
        **kwds: Any) -> Any:
    ...


def lag_plot(
        series: Any, lag: int = ..., ax: Optional[Any] = ...,
        **kwds: Any) -> Any:
    ...


def autocorrelation_plot(
        series: Any, ax: Optional[Any] = ..., **kwds: Any) -> Any:
    ...


def tsplot(
        series: Any, plotf: Any, ax: Optional[Any] = ...,
        **kwargs: Any) -> Any:
    ...


class _Options(dict):
    def __init__(self, deprecated: bool = ...) -> None:
        ...

    def __getitem__(self, key: Any) -> Any:
        ...

    def __setitem__(self, key: Any, value: Any) -> Any:
        ...

    def __delitem__(self, key: Any) -> Any:
        ...

    def __contains__(self, key: Any) -> Any:
        ...

    def reset(self) -> None:
        ...

    def use(self, key: Any, value: Any) -> None:
        ...


plot_params: Any
