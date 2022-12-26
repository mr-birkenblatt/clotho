from typing import Dict, List

from ..package_exporter import PackagingError as PackagingError


def find_first_use_of_broken_modules(exc: PackagingError) -> Dict[str, List[str]]: ...
