def DataPipeToQueuesLoop(source_datapipe, req_queue, res_queue) -> None: ...
def SpawnProcessForDataPipeline(multiprocessing_ctx, datapipe): ...
def SpawnThreadForDataPipeline(datapipe): ...