from abc import ABC, abstractmethod
from typing import List, Iterable
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
            result.append([word.lower() for word in lemmas])
        return result


class CombinedEmbedding(EmbeddingBase):

    def __init__(self, models: Iterable[EmbeddingBase]):
        self.models = models
        self.size = sum([model.dim() for model in self.models])

    def prepare(self, params, samples: List[List[str]]):
        for model in self.models:
            model.prepare(params, samples)

    def batcher(self, params, batch: List[List[str]]) -> np.ndarray:
        results: List[np.ndarray] = [model.batcher(params, batch) for model in self.models]
        return np.hstack(results)

    def dim(self) -> int:
        return self.size
