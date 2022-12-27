# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access,no-name-in-module,undefined-variable


torch_parent: str


def get_file_path(*path_components: str) -> str: ...


def get_file_path_2(*path_components: str) -> str: ...


def get_writable_path(path: str) -> str: ...


def prepare_multiprocessing_environment(path: str) -> None: ...


def resolve_library_path(path: str) -> str: ...


TEST_MASTER_ADDR: str
TEST_MASTER_PORT: int
USE_GLOBAL_DEPS: bool
USE_RTLD_GLOBAL_WITH_LIBTORCH: bool
