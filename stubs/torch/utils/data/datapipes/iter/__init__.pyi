# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from torch.utils.data.datapipes.iter.callable import (
    CollatorIterDataPipe as Collator,
)
from torch.utils.data.datapipes.iter.callable import (
    MapperIterDataPipe as Mapper,
)
from torch.utils.data.datapipes.iter.combinatorics import (
    SamplerIterDataPipe as Sampler,
)
from torch.utils.data.datapipes.iter.combinatorics import (
    ShufflerIterDataPipe as Shuffler,
)
from torch.utils.data.datapipes.iter.combining import (
    ConcaterIterDataPipe as Concater,
)
from torch.utils.data.datapipes.iter.combining import (
    DemultiplexerIterDataPipe as Demultiplexer,
)
from torch.utils.data.datapipes.iter.combining import (
    ForkerIterDataPipe as Forker,
)
from torch.utils.data.datapipes.iter.combining import (
    MultiplexerIterDataPipe as Multiplexer,
)
from torch.utils.data.datapipes.iter.combining import (
    ZipperIterDataPipe as Zipper,
)
from torch.utils.data.datapipes.iter.filelister import (
    FileListerIterDataPipe as FileLister,
)
from torch.utils.data.datapipes.iter.fileopener import (
    FileLoaderIterDataPipe as FileLoader,
)
from torch.utils.data.datapipes.iter.fileopener import (
    FileOpenerIterDataPipe as FileOpener,
)
from torch.utils.data.datapipes.iter.grouping import (
    BatcherIterDataPipe as Batcher,
)
from torch.utils.data.datapipes.iter.grouping import (
    GrouperIterDataPipe as Grouper,
)
from torch.utils.data.datapipes.iter.grouping import (
    ShardingFilterIterDataPipe as ShardingFilter,
)
from torch.utils.data.datapipes.iter.grouping import (
    UnBatcherIterDataPipe as UnBatcher,
)
from torch.utils.data.datapipes.iter.routeddecoder import (
    RoutedDecoderIterDataPipe as RoutedDecoder,
)
from torch.utils.data.datapipes.iter.selecting import (
    FilterIterDataPipe as Filter,
)
from torch.utils.data.datapipes.iter.streamreader import (
    StreamReaderIterDataPipe as StreamReader,
)
from torch.utils.data.datapipes.iter.utils import (
    IterableWrapperIterDataPipe as IterableWrapper,
)
