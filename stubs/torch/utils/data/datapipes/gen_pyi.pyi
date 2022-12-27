# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors
from typing import Any, Dict, List, Set, Tuple, Union


def materialize_lines(lines: List[str], indentation: int) -> str: ...


def gen_from_template(
    dir: str, template_name: str, output_name: str,
    replacements: List[Tuple[str, Any, int]]): ...


def find_file_paths(
    dir_paths: List[str], files_to_exclude: Set[str]) -> Set[str]: ...


def extract_method_name(line: str) -> str: ...


def extract_class_name(line: str) -> str: ...


def parse_datapipe_file(
    file_path: str) -> Tuple[Dict[str, str], Dict[str, str], Set[str]]: ...


def parse_datapipe_files(
    file_paths: Set[str],
    ) -> Tuple[Dict[str, str], Dict[str, str], Set[str]]: ...


def split_outside_bracket(
    line: str, delimiter: str = ...) -> List[str]: ...


def process_signature(line: str) -> str: ...


def get_method_definitions(
    file_path: Union[str, List[str]], files_to_exclude: Set[str],
    deprecated_files: Set[str], default_output_type: str,
    method_to_special_output_type: Dict[str, str],
    root: str = ...) -> List[str]: ...


iterDP_file_path: str
iterDP_files_to_exclude: Set[str]
iterDP_deprecated_files: Set[str]
iterDP_method_to_special_output_type: Dict[str, str]
mapDP_file_path: str
mapDP_files_to_exclude: Set[str]
mapDP_deprecated_files: Set[str]
mapDP_method_to_special_output_type: Dict[str, str]


def main() -> None: ...
