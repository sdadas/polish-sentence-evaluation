from pathlib import Path
from typing import Dict

import numpy as np
from allennlp.commands.elmo import ElmoEmbedder
from flair.data import Sentence
from flair.embeddings import BertEmbeddings, FlairEmbeddings

from data import Sent
from embeddings.base import EmbeddingBase
from gensim.models import KeyedVectors


class KeyedVectorsEmbedding(EmbeddingBase):

    def __init__(self, path: Path):
        self.embedding: KeyedVectors = self.__load_embedding(path)
        self.__size: int = self.embedding.wv.vector_size

    def __load_embedding(self, path: Path, text_format: bool=False) -> KeyedVectors:
        infile: str = str(path.absolute())
        return KeyedVectors.load(infile) if not text_format else KeyedVectors.load_word2vec_format(infile, binary=False)

    def __vocab_index(self, word: str) -> int:
        return self.embedding.wv.vocab.get(word, self.embedding.wv.vocab['<unk>']).index

    def __vocab_vector(self, word: str) -> np.ndarray:
        return self.embedding.wv.syn0[self.__vocab_index(word)]

    def embed(self, sentence: Sent) -> np.ndarray:
        res: np.ndarray = np.zeros(self.__size, dtype=np.float32)
        for word in sentence.lemmas:
            vec = self.__vocab_vector(word)
            res += vec
        return res

    def dim(self) -> int:
        return self.__size


class RandomEmbedding(EmbeddingBase):

    def __init__(self):
        self.words: Dict[str, np.ndarray] = {}
        self.__size = 100

    def embed(self, sentence: Sent) -> np.ndarray:
        res: np.ndarray = np.zeros(self.__size, dtype=np.float32)
        for word in sentence.lemmas:
            vec = self.__vocab_vector(word)
            res += vec
        return res

    def __vocab_vector(self, word: str) -> np.ndarray:
        res = self.words.get(word, None)
        if res is None:
            res = np.random.rand(self.__size)
            res = res / np.linalg.norm(res, ord=2)
            self.words[word] = res
        return res

    def dim(self) -> int:
        return self.__size


class BertEmbedding(EmbeddingBase):

    def __init__(self):
        self.model = BertEmbeddings(bert_model_or_path="bert-base-multilingual-cased")
        self.size = 3072

    def embed(self, sentence: Sent) -> np.ndarray:
        outputs = self.model.embed(Sentence(" ".join(sentence.tokens)))
        out = outputs[0]
        res = np.zeros(self.size, dtype=np.float32)
        for token in out.tokens:
            res += np.fromiter(token.embedding.tolist(), dtype=np.float32)
        return res

    def dim(self) -> int:
        return self.size


class ElmoEmbedding(EmbeddingBase):

    def __init__(self, path: Path, layers:str="all"):
        assert layers == "all" or layers == "top"
        options: str = str(path / "options.json")
        weights: str = str(path / "weights.hdf5")
        self.model: ElmoEmbedder = ElmoEmbedder(options, weights)
        self.layers: str = layers
        self.vector_size = self.__get_vector_size()

    def __get_vector_size(self) -> int:
        vectors = self.model.embed_sentence(["test"])
        return len(vectors[0][0])

    def embed(self, sentence: Sent) -> np.ndarray:
        vectors = self.model.embed_sentence(sentence.tokens)
        tokens_num = len(vectors[0])
        res: np.ndarray = np.zeros(self.dim(), dtype=np.float32)
        for idx in range(tokens_num):
            if self.layers == "top":
                vector = vectors[2][idx]
            elif self.layers == "all":
                vector = np.hstack((vectors[0][idx], vectors[1][idx], vectors[2][idx]))
            else: raise AssertionError("Unknown 'layers' parameter")
            res += vector
        return res

    def dim(self) -> int:
        return self.vector_size if self.layers == "top" else 3 * self.vector_size


class FlairEmbedding(EmbeddingBase):

    def __init__(self):
        self.forward_model = FlairEmbeddings("pl-forward")
        self.backward_model = FlairEmbeddings("pl-backward")
        self.size = 4096

    def embed(self, sentence: Sent) -> np.ndarray:
        outputs_forward = self.forward_model.embed(Sentence(" ".join(sentence.tokens)))
        outputs_backward = self.backward_model.embed(Sentence(" ".join(sentence.tokens)))
        res = np.zeros(self.size, dtype=np.float32)
        for idx in range(len(sentence.tokens)):
            out_fwd = np.fromiter(outputs_forward[0].tokens[idx].embedding.tolist(), dtype=np.float32)
            out_bwd = np.fromiter(outputs_backward[0].tokens[idx].embedding.tolist(), dtype=np.float32)
            out = np.hstack((out_fwd, out_bwd))
            res += out
        return res

    def dim(self) -> int:
        return self.size
