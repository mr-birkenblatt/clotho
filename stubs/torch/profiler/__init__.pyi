# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access,no-name-in-module,undefined-variable


from torch.autograd import DeviceType as DeviceType
from torch.autograd import kineto_available as kineto_available
from torch.autograd.profiler import record_function as record_function

from .profiler import profile as profile
from .profiler import ProfilerAction as ProfilerAction
from .profiler import ProfilerActivity as ProfilerActivity
from .profiler import schedule as schedule
from .profiler import supported_activities as supported_activities
from .profiler import tensorboard_trace_handler as tensorboard_trace_handler
