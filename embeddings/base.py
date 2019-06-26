from abc import ABC, abstractmethod

import numpy as np

from data import Sent, Corpus


class EmbeddingBase(ABC):

    @abstractmethod
    def embed(self, sentence: Sent) -> np.ndarray: raise NotImplementedError

    @abstractmethod
    def dim(self) -> int: raise NotImplementedError

    def embed_corpus(self, corpus: Corpus, output_path: str) -> np.ndarray:
        rows: int = len(corpus)
        cols: int = corpus.sentences_per_sample
        result = np.ndarray((rows, cols, self.dim()), dtype=np.float32)
        idx = 0
        for sample in corpus.samples:
            for col_idx, sent in enumerate(sample):
                result[idx, col_idx, :] = self.embed(sent)
        if output_path is not None:
            np.save(output_path, result)
        return result