import logging
from pathlib import Path
from typing import List, Dict
import numpy as np
from fse.models import Sentence2Vec

from gensim.models import KeyedVectors

from methods.base import EmbeddingBase


class SIFEmbedding(EmbeddingBase):

    def __init__(self, path: Path):
        self.embedding: KeyedVectors = self._load_embedding(path)
        self.model = Sentence2Vec(self.embedding, lang="pl")
        self._size: int = self.embedding.wv.vector_size
        self.cache: Dict[str, np.ndarray] = {}

    def _load_embedding(self, path: Path) -> KeyedVectors:
        text_format: bool = path.name.endswith(".txt")
        infile: str = str(path.absolute())
        return KeyedVectors.load(infile) if not text_format else KeyedVectors.load_word2vec_format(infile, binary=False)

    def prepare(self, params, samples: List[List[str]]):
        logging.info("Preparing %d sentences", len(samples))
        if params.lemmatize: samples = self.lemmatize(params, samples)
        result: np.ndarray = self.model.train(samples)
        for idx, sample in enumerate(samples):
            self.cache[" ".join(sample)] = result[idx, :]

    def batcher(self, params, batch: List[List[str]]) -> np.ndarray:
        if params.lemmatize: batch = self.lemmatize(params, batch)
        embeddings = []
        for sent in batch: embeddings.append(self.embed(sent))
        embeddings = np.vstack(embeddings)
        return embeddings

    def embed(self, sentence: List[str]):
        key = " ".join(sentence)
        return self.cache[key]

    def dim(self) -> int:
        return self._size