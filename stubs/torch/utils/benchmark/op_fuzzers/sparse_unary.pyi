# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from torch.utils.benchmark import FuzzedParameter as FuzzedParameter


        FuzzedSparseTensor as FuzzedSparseTensor, Fuzzer as Fuzzer,
        ParameterAlias as ParameterAlias


class UnaryOpSparseFuzzer(Fuzzer):
    def __init__(self, seed, dtype=..., cuda: bool = ...) -> None: ...
