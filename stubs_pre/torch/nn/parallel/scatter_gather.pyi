# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access,no-name-in-module,undefined-variable


from ._functions import Gather as Gather
from ._functions import Scatter as Scatter


def is_namedtuple(obj): ...


def scatter(inputs, target_gpus, dim: int = ...): ...


def scatter_kwargs(inputs, kwargs, target_gpus, dim: int = ...): ...


def gather(outputs, target_device, dim: int = ...): ...
