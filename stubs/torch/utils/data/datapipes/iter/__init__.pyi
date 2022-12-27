# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from torch.utils.data.datapipes.iter.callable import as, CollatorIterDataPipe


        Collator, MapperIterDataPipe as Mapper
from torch.utils.data.datapipes.iter.combinatorics import
        SamplerIterDataPipe as Sampler, ShufflerIterDataPipe as Shuffler
from torch.utils.data.datapipes.iter.combining import ConcaterIterDataPipe


        as Concater, DemultiplexerIterDataPipe as Demultiplexer,
        ForkerIterDataPipe as Forker, MultiplexerIterDataPipe as Multiplexer,
        ZipperIterDataPipe as Zipper
from torch.utils.data.datapipes.iter.filelister import
        FileListerIterDataPipe as FileLister
from torch.utils.data.datapipes.iter.fileopener import
        FileLoaderIterDataPipe as FileLoader,
        FileOpenerIterDataPipe as FileOpener
from torch.utils.data.datapipes.iter.grouping import as, BatcherIterDataPipe


        Batcher, GrouperIterDataPipe as Grouper,
        ShardingFilterIterDataPipe as ShardingFilter,
        UnBatcherIterDataPipe as UnBatcher
from torch.utils.data.datapipes.iter.routeddecoder import
        RoutedDecoderIterDataPipe as RoutedDecoder
from torch.utils.data.datapipes.iter.selecting import as, FilterIterDataPipe


        Filter
from torch.utils.data.datapipes.iter.streamreader import
        StreamReaderIterDataPipe as StreamReader
from torch.utils.data.datapipes.iter.utils import
        IterableWrapperIterDataPipe as IterableWrapper
