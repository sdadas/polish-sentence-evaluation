import math
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Optional

import numpy as np

from methods.base import EmbeddingBase


class VectorFeature(ABC):

    def __init__(self, parent: EmbeddingBase):
        self.parent = parent

    @abstractmethod
    def apply(self, pos: int, word: str, weight: float, vector) -> Tuple[float, Optional[np.ndarray]]:
        raise NotImplementedError


class RandomUnkFeature(VectorFeature):

    def __init__(self, parent: EmbeddingBase):
        super().__init__(parent)
        self.words: Dict[str, np.ndarray] = {}

    def apply(self, pos: int, word: str, weight: float, vector) -> Tuple[float, Optional[np.ndarray]]:
        if vector is not None: return (weight, vector)
        else: return (weight, self._vocab_vector(word))

    def _vocab_vector(self, word: str) -> np.ndarray:
        res = self.words.get(word, None)
        if res is None:
            res = np.random.rand(self.parent.dim())
            res = res / np.linalg.norm(res, ord=2)
            self.words[word] = res
        return res


class PositionalEncodingFeature(VectorFeature):

    def __init__(self, parent: EmbeddingBase):
        super().__init__(parent)
        self.encodings: Dict[int, np.ndarray] = {}

    def apply(self, pos: int, word: str, weight: float, vector) -> Tuple[float, Optional[np.ndarray]]:
        if vector is None: return (weight, vector)
        else: return (weight, vector + self._get_encoding(pos + 1))

    def _get_encoding(self, pos: int):
        res = self.encodings.get(pos)
        size = self.parent.dim()
        if res is None:
            values = [float(pos) / math.pow(10000.0, 2.0*math.floor(0.5*float(i)) / 512.0) for i in range(size)]
            values = [math.sin(val) if i % 2 == 0 else math.cos(val) for i, val in enumerate(values)]
            res = np.array(values)
            res /= 10
            self.encodings[pos] = res
        return res


class SIFFeature(VectorFeature):

    def __init__(self, parent: EmbeddingBase):
        super().__init__(parent)
        self.sif = self._precompute_sif_weights(parent.embedding)

    def _precompute_sif_weights(self, wv, alpha=1e-3):
        corpus_size = 0
        sif = np.zeros(shape=len(wv.vocab), dtype=np.float)
        for k in wv.index2word: corpus_size += wv.vocab[k].count
        for idx, k in enumerate(wv.index2word):
            pw = wv.vocab[k].count / corpus_size
            sif[idx] = alpha / (alpha+pw)
        return sif

    def apply(self, pos: int, word: str, weight: float, vector) -> Tuple[float, Optional[np.ndarray]]:
        if vector is None: return (weight, vector)
        else: return (self._get_weight(word), vector)

    def _get_weight(self, word: str):
        idx = self.parent.embedding.vocab[word].index
        return self.sif[idx]
