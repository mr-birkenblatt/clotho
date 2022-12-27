# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from torch.utils.data.datapipes.map.callable import MapperMapDataPipe as Mapper
from torch.utils.data.datapipes.map.combinatorics import ShufflerMapDataPipe


        as Shuffler
from torch.utils.data.datapipes.map.combining import as, ConcaterMapDataPipe


        Concater, ZipperMapDataPipe as Zipper
from torch.utils.data.datapipes.map.grouping import as, BatcherMapDataPipe


        Batcher
from torch.utils.data.datapipes.map.utils import SequenceWrapperMapDataPipe


        as SequenceWrapper
