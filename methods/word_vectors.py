from pathlib import Path
from typing import List, Optional, Dict

from allennlp.commands.elmo import ElmoEmbedder
from flair.data import Sentence
from flair.embeddings import BertEmbeddings, FlairEmbeddings
from gensim.models import KeyedVectors
from methods.base import EmbeddingBase
import numpy as np


class KeyedVectorsEmbedding(EmbeddingBase):

    def __init__(self, path: Path, pooling: str="avg"):
        assert pooling in ("avg", "max", "concat")
        self.embedding: KeyedVectors = self._load_embedding(path)
        self._size: int = self.embedding.wv.vector_size
        self.pooling = pooling
        self.pooling_op = {"avg": self.avg_pool, "max": self.max_pool, "concat": self.concat_pool}[self.pooling]

    def _load_embedding(self, path: Path) -> KeyedVectors:
        text_format: bool = path.name.endswith(".txt")
        infile: str = str(path.absolute())
        return KeyedVectors.load(infile) if not text_format else KeyedVectors.load_word2vec_format(infile, binary=False)

    def _vocab_index(self, word: str) -> int:
        voc = self.embedding.wv.vocab.get(word)
        return voc.index if voc else -1

    def _vocab_vector(self, word: str) -> Optional[np.ndarray]:
        idx = self._vocab_index(word)
        if idx < 0: return None
        vec = self.embedding.wv.syn0[idx]
        return vec / np.linalg.norm(vec)

    def embed(self, sentence: List[str]):
        sentvec = [self._vocab_vector(word) for word in sentence]
        sentvec = [vec for vec in sentvec if vec is not None]
        if not sentvec:
            vec = np.zeros(self._size)
            sentvec.append(vec)
        sentvec = self.pooling_op(sentvec)
        return sentvec

    def prepare(self, params, samples: List[str]):
        pass

    def dim(self) -> int:
        return self._size * 2 if self.pooling == "concat" else self._size

    def avg_pool(self, sentvec):
        return np.mean(sentvec, 0)

    def max_pool(self, sentvec):
        return np.max(sentvec, 0)

    def concat_pool(self, sentvec):
        return np.hstack((self.avg_pool(sentvec), self.max_pool(sentvec)))


class RandomEmbedding(EmbeddingBase):

    def __init__(self):
        self.words: Dict[str, np.ndarray] = {}
        self._size = 100

    def _vocab_vector(self, word: str) -> np.ndarray:
        res = self.words.get(word, None)
        if res is None:
            res = np.random.rand(self._size)
            res = res / np.linalg.norm(res, ord=2)
            self.words[word] = res
        return res

    def embed(self, sentence: List[str]):
        sentvec = [self._vocab_vector(word) for word in sentence]
        sentvec = np.mean(sentvec, 0)
        return sentvec

    def dim(self) -> int:
        return self._size


class BertEmbedding(EmbeddingBase):

    def __init__(self):
        self.model = BertEmbeddings(bert_model_or_path="bert-base-multilingual-cased")
        self.size = 3072

    def _get_vector(self, sentence: Sentence) -> np.ndarray:
        res = np.zeros(self.size, dtype=np.float32)
        for token in sentence.tokens:
            vec = np.fromiter(token.embedding.tolist(), dtype=np.float32)
            vec = vec / np.linalg.norm(vec, ord=2)
            res += vec
        res /= len(sentence.tokens)
        return res

    def batcher(self, params, batch: List[List[str]]) -> np.ndarray:
        batch = [Sentence(" ".join(sent)) if sent != [] else ['.'] for sent in batch]
        embeddings = []
        sentences = self.model.embed(batch)
        for sent in sentences:
            embeddings.append(self._get_vector(sent))
        embeddings = np.vstack(embeddings)
        return embeddings

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

    def _get_vector(self, vectors) -> np.ndarray:
        tokens_num = len(vectors[0])
        res: np.ndarray = np.zeros(self.dim(), dtype=np.float32)
        for idx in range(tokens_num):
            if self.layers == "top":
                vector = vectors[2][idx]
            elif self.layers == "all":
                vector = np.hstack((vectors[0][idx], vectors[1][idx], vectors[2][idx]))
            else: raise AssertionError("Unknown 'layers' parameter")
            res += vector
        res /= tokens_num
        return res

    def batcher(self, params, batch: List[List[str]]) -> np.ndarray:
        batch = [sent if sent != [] else ['.'] for sent in batch]
        embeddings = []
        vectors = self.model.embed_sentences(batch, batch_size=len(batch))
        for vec in vectors:
            embeddings.append(self._get_vector(vec))
        embeddings = np.vstack(embeddings)
        return embeddings

    def dim(self) -> int:
        return self.vector_size if self.layers == "top" else 3 * self.vector_size


class FlairEmbedding(EmbeddingBase):

    def __init__(self):
        self.forward_model = FlairEmbeddings("pl-forward")
        self.backward_model = FlairEmbeddings("pl-backward")
        self.size = 8192

    def _get_vector(self, forward: Sentence, backward: Sentence) -> np.ndarray:
        res = np.zeros(self.size, dtype=np.float32)
        for idx in range(len(forward)):
            out_fwd = np.fromiter(forward.tokens[idx].embedding.tolist(), dtype=np.float32)
            out_bwd = np.fromiter(backward.tokens[idx].embedding.tolist(), dtype=np.float32)
            out = np.hstack((out_fwd, out_bwd))
            res += out
        res /= len(forward)
        return res

    def batcher(self, params, batch: List[List[str]]) -> np.ndarray:
        batch = [Sentence(" ".join(sent)) if sent != [] else ['.'] for sent in batch]
        embeddings = []
        outputs_forward = self.forward_model.embed(batch)
        outputs_backward = self.backward_model.embed(batch)
        for forward, backward in zip(outputs_forward, outputs_backward):
            embeddings.append(self._get_vector(forward, backward))
        embeddings = np.vstack(embeddings)
        return embeddings

    def dim(self) -> int:
        return self.size
