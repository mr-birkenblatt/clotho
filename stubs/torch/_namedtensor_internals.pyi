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
