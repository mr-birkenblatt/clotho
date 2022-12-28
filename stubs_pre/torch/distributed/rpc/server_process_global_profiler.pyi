# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access,no-name-in-module,undefined-variable


from _typeshed import Incomplete
from torch.autograd.profiler_legacy import profile as profile


class _server_process_global_profile(profile):
    def __init__(self, *args, **kwargs) -> None: ...
    entered: bool
    def __enter__(self): ...
    function_events: Incomplete
    process_global_function_events: Incomplete
    def __exit__(self, exc_type, exc_val, exc_tb): ...
