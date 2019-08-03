from abc import ABC, abstractmethod
from typing import List
import numpy as np


class EmbeddingBase(ABC):

    @abstractmethod
    def dim(self) -> int:
        raise NotImplementedError

    def prepare(self, params, samples: List[List[str]]):
        pass

    def embed(self, sentence: List[str]):
        raise NotImplementedError

    def batcher(self, params, batch: List[List[str]]) -> np.ndarray:
        if params.lemmatize: batch = self.lemmatize(params, batch)
        batch = [sent if sent != [] else ['.'] for sent in batch]
        embeddings = []
        for sent in batch: embeddings.append(self.embed(sent))
        embeddings = np.vstack(embeddings)
        return embeddings

    def lemmatize(self, params, batch: List[List[str]]) -> List[List[str]]:
        if not params.analyzer: return batch
        result = []
        for sent in batch:
            _, lemmas = params.analyzer.analyze(" ".join(sent))
            result.append(lemmas)
        return result
