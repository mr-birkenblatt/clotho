from torch.utils.data.datapipes.datapipe import DataChunk


class DataChunkDF(DataChunk):
    def __iter__(self): ...
    def __len__(self): ...
