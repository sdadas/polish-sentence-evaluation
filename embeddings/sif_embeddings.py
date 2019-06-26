from fse.models import Sentence2Vec
from gensim.models import KeyedVectors

from data import Corpus, Sent
from embeddings.base import EmbeddingBase
import numpy as np


class SIFEmbedding(EmbeddingBase):

    def __init__(self, word_embeddings: KeyedVectors):
        self.model = Sentence2Vec(word_embeddings, lang="pl")
        self.__size: int = word_embeddings.wv.vector_size

    def embed(self, sentence: Sent) -> np.ndarray: raise NotImplementedError

    def dim(self) -> int:
        return self.__size

    def embed_corpus(self, corpus: Corpus, output_path: str) -> np.ndarray:
        sentences = [sent.lemmas for sample in corpus.samples for sent in sample]
        result: np.ndarray = self.model.train(sentences)
        if output_path is not None:
            np.save(output_path, result)
        return result
