# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access,no-name-in-module,undefined-variable


class FauxTorch:
    def __init__(self, real_torch, extra_ns_per_element) -> None: ...
    def extra_overhead(self, result): ...
    def add(self, *args, **kwargs): ...
    def mul(self, *args, **kwargs): ...
    def cat(self, *args, **kwargs): ...
    def matmul(self, *args, **kwargs): ...


def main() -> None: ...
