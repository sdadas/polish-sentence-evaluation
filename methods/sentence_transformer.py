from typing import List
import numpy as np

from sentence_transformers import SentenceTransformer

from methods.base import EmbeddingBase


class SentenceTransformersEmbedding(EmbeddingBase):

    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.encode(["test"]).shape[1]

    def dim(self) -> int:
        return self.dim

    def batcher(self, params, batch: List[List[str]]) -> np.ndarray:
        batch = [" ".join(sent) if sent != [] else "." for sent in batch]
        embeddings = self.model.encode(batch, show_progress_bar=False)
        return embeddings
