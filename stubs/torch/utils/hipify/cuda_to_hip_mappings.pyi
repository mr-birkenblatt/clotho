# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from .constants import API_BLAS as API_BLAS
from .constants import API_C10 as API_C10


        API_CAFFE2 as API_CAFFE2, API_DRIVER as API_DRIVER,
        API_FFT as API_FFT, API_PYTORCH as API_PYTORCH, API_RAND as API_RAND,
        API_ROCTX as API_ROCTX, API_RTC as API_RTC,
        API_RUNTIME as API_RUNTIME, API_SPARSE as API_SPARSE,
        CONV_CACHE as CONV_CACHE, CONV_CONTEXT as CONV_CONTEXT,
        CONV_D3D10 as CONV_D3D10, CONV_D3D11 as CONV_D3D11,
        CONV_D3D9 as CONV_D3D9, CONV_DEF as CONV_DEF,
        CONV_DEVICE as CONV_DEVICE, CONV_DEVICE_FUNC as CONV_DEVICE_FUNC,
        CONV_EGL as CONV_EGL, CONV_ERROR as CONV_ERROR,
        CONV_EVENT as CONV_EVENT, CONV_EXEC as CONV_EXEC, CONV_GL as CONV_GL,
        CONV_GRAPHICS as CONV_GRAPHICS, CONV_INCLUDE as CONV_INCLUDE,
        CONV_INCLUDE_CUDA_MAIN_H as CONV_INCLUDE_CUDA_MAIN_H,
        CONV_INIT as CONV_INIT, CONV_JIT as CONV_JIT,
        CONV_MATH_FUNC as CONV_MATH_FUNC, CONV_MEM as CONV_MEM,
        CONV_MODULE as CONV_MODULE,
        CONV_NUMERIC_LITERAL as CONV_NUMERIC_LITERAL,
        CONV_OCCUPANCY as CONV_OCCUPANCY, CONV_OTHER as CONV_OTHER,
        CONV_PEER as CONV_PEER, CONV_SPECIAL_FUNC as CONV_SPECIAL_FUNC,
        CONV_STREAM as CONV_STREAM, CONV_SURFACE as CONV_SURFACE,
        CONV_TEX as CONV_TEX, CONV_THREAD as CONV_THREAD,
        CONV_TYPE as CONV_TYPE, CONV_VDPAU as CONV_VDPAU,
        CONV_VERSION as CONV_VERSION, HIP_UNSUPPORTED as HIP_UNSUPPORTED
from _typeshed import Incomplete


MATH_TRANSPILATIONS: Incomplete
CUDA_TYPE_NAME_MAP: Incomplete
CUDA_INCLUDE_MAP: Incomplete
CUDA_IDENTIFIER_MAP: Incomplete
CUDA_SPARSE_MAP: Incomplete
PYTORCH_SPECIFIC_MAPPINGS: Incomplete
CAFFE2_SPECIFIC_MAPPINGS: Incomplete
C10_MAPPINGS: Incomplete
CUDA_TO_HIP_MAPPINGS: Incomplete
