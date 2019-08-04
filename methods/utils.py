from pathlib import Path
from typing import Callable
import numpy as np
import logging

def cached(batcher_func: Callable, cache: Path) -> Callable:
    def batcher(params, batch):
        task_name = params.current_task
        dataset = params.get("batcher_dataset")
        offset = params.get("batcher_offset")
        size = str(len(batch))
        if task_name and dataset and offset:
            file_name = f"{task_name}_{dataset}_{offset}_{size}.npy"
            cached_file = cache / file_name
            if cached_file.exists():
                logging.debug("Loading batch from cache %s", file_name)
                return np.load(str(cached_file.absolute()))
            else:
                embeddings = batcher_func(params, batch)
                logging.debug("Saving batch to cache %s", file_name)
                np.save(str(cached_file.absolute()), embeddings)
                return embeddings
        else:
            return batcher_func(params, batch)

    return batcher