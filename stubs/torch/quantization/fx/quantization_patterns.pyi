# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from torch.ao.quantization.fx.quantization_patterns import (
    BatchNormQuantizeHandler as BatchNormQuantizeHandler,
)
from torch.ao.quantization.fx.quantization_patterns import (
    BinaryOpQuantizeHandler as BinaryOpQuantizeHandler,
)
from torch.ao.quantization.fx.quantization_patterns import (
    CatQuantizeHandler as CatQuantizeHandler,
)
from torch.ao.quantization.fx.quantization_patterns import (
    ConvReluQuantizeHandler as ConvReluQuantizeHandler,
)
from torch.ao.quantization.fx.quantization_patterns import (
    CopyNodeQuantizeHandler as CopyNodeQuantizeHandler,
)
from torch.ao.quantization.fx.quantization_patterns import (
    CustomModuleQuantizeHandler as CustomModuleQuantizeHandler,
)
from torch.ao.quantization.fx.quantization_patterns import (
    DefaultNodeQuantizeHandler as DefaultNodeQuantizeHandler,
)
from torch.ao.quantization.fx.quantization_patterns import (
    EmbeddingQuantizeHandler as EmbeddingQuantizeHandler,
)
from torch.ao.quantization.fx.quantization_patterns import (
    FixedQParamsOpQuantizeHandler as FixedQParamsOpQuantizeHandler,
)
from torch.ao.quantization.fx.quantization_patterns import (
    GeneralTensorShapeOpQuantizeHandler as GeneralTensorShapeOpQuantizeHandler,
)
from torch.ao.quantization.fx.quantization_patterns import (
    LinearReLUQuantizeHandler as LinearReLUQuantizeHandler,
)
from torch.ao.quantization.fx.quantization_patterns import (
    QuantizeHandler as QuantizeHandler,
)
from torch.ao.quantization.fx.quantization_patterns import (
    RNNDynamicQuantizeHandler as RNNDynamicQuantizeHandler,
)
from torch.ao.quantization.fx.quantization_patterns import (
    StandaloneModuleQuantizeHandler as StandaloneModuleQuantizeHandler,
)
