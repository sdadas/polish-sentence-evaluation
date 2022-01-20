from typing import List

import tensorflow_hub as hub
import numpy as np
import tensorflow_text

from methods.base import EmbeddingBase


class USEEmbedding(EmbeddingBase):

    def __init__(self):
        self.embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3")

    def dim(self) -> int:
        return 512

    def batcher(self, params, batch: List[List[str]]) -> np.ndarray:
        batch = [" ".join(sent) if sent != [] else ['.'] for sent in batch]
        embeddings = self.embed(batch)
        embeddings = np.vstack(embeddings)
        return embeddings
