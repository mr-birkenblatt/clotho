# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access,no-name-in-module,undefined-variable


class MetaInitErrorInfo:
    mode_name: str
    mode_class_name: str
    def __init__(self, mode_name, mode_class_name) -> None: ...


class _ModeInfo:
    mode_name: str
    mode_class: type
    base_mode_class: type
    def mode_class_name(self): ...
    def get_mode(self) -> None: ...
    def set_mode(self, mode) -> None: ...
    def __init__(self, mode_name, mode_class, base_mode_class) -> None: ...
