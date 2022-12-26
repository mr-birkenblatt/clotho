from torch.autograd import DeviceType as DeviceType
from torch.autograd import kineto_available as kineto_available
from torch.autograd.profiler import record_function as record_function

from .profiler import profile as profile
from .profiler import ProfilerAction as ProfilerAction
from .profiler import ProfilerActivity as ProfilerActivity
from .profiler import schedule as schedule
from .profiler import supported_activities as supported_activities
from .profiler import tensorboard_trace_handler as tensorboard_trace_handler
