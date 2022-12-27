# pylint: disable=multiple-statements,unused-argument, invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias, unused-import
# pylint: disable=redefined-builtin,super-init-not-called, arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors, import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member, keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name, c-extension-no-member
# pylint: disable=protected-access,no-name-in-module, undefined-variable


def check_serializing_named_tensor(tensor) -> None: ...


def build_dim_map(tensor): ...


def unzip_namedshape(namedshape): ...


def namer_api_name(inplace): ...


def is_ellipsis(item): ...


def single_ellipsis_index(names, fn_name): ...


def expand_single_ellipsis(numel_pre_glob, numel_post_glob, names): ...


def replace_ellipsis_by_position(ellipsis_idx, names, tensor_names): ...


def resolve_ellipsis(names, tensor_names, fn_name): ...


def update_names_with_list(tensor, names, inplace): ...


def update_names_with_mapping(tensor, rename_map, inplace): ...


def update_names(tensor, names, rename_map, inplace): ...
