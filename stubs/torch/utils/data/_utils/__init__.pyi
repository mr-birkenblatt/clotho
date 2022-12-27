# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from _typeshed import Incomplete
from torch._utils import ExceptionWrapper as ExceptionWrapper

from . import collate as collate
from . import fetch as fetch
from . import pin_memory as pin_memory
from . import signal_handling as signal_handling
from . import worker as worker


IS_WINDOWS: Incomplete
MP_STATUS_CHECK_INTERVAL: float
python_exit_status: bool
DATAPIPE_SHARED_SEED: str
DATAPIPE_SHARED_SEED_COUNTER: str
DATAPIPE_SHARED_SEED_DEFAULT_TIMEOUT: Incomplete
DATAPIPE_SHARED_SEED_CHECK_INTERVAL: float
HAS_NUMPY: bool
