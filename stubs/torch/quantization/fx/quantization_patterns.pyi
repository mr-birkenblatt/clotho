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


        BinaryOpQuantizeHandler as BinaryOpQuantizeHandler,
        CatQuantizeHandler as CatQuantizeHandler,
        ConvReluQuantizeHandler as ConvReluQuantizeHandler,
        CopyNodeQuantizeHandler as CopyNodeQuantizeHandler,
        CustomModuleQuantizeHandler as CustomModuleQuantizeHandler,
        DefaultNodeQuantizeHandler as DefaultNodeQuantizeHandler,
        EmbeddingQuantizeHandler as EmbeddingQuantizeHandler,
        FixedQParamsOpQuantizeHandler as FixedQParamsOpQuantizeHandler,
        GeneralTensorShapeOpQuantizeHandler as \
        GeneralTensorShapeOpQuantizeHandler,
        LinearReLUQuantizeHandler as LinearReLUQuantizeHandler,
        QuantizeHandler as QuantizeHandler,
        RNNDynamicQuantizeHandler as RNNDynamicQuantizeHandler,
        StandaloneModuleQuantizeHandler as StandaloneModuleQuantizeHandler
