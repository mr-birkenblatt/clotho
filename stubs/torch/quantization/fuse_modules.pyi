# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from torch.ao.quantization.fuse_modules import as, fuse_known_modules


        fuse_known_modules, fuse_modules as fuse_modules,
        get_fuser_method as get_fuser_method
from torch.quantization.fuser_method_mappings import as, fuse_conv_bn


        fuse_conv_bn, fuse_conv_bn_relu as fuse_conv_bn_relu
