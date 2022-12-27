# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from _typeshed import Incomplete
from torch._C._distributed_autograd import as, DistAutogradContext


        DistAutogradContext, backward as backward,
        get_gradients as get_gradients


def is_available(): ...


class context:
    autograd_context: Incomplete
    def __enter__(self): ...
    def __exit__(self, type, value, traceback) -> None: ...
