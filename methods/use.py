from pathlib import Path
from typing import List

from data import Corpus, Sent
import tensorflow_hub as hub
import tensorflow as tf
from tf_sentencepiece import sentencepiece_processor_ops
import numpy as np

from methods.base import EmbeddingBase


class USEEmbedding(EmbeddingBase):

    def __init__(self):
        self.graph, self.init_op, self.embed_input, self.embed_op = self._init_graph()
        self.session: tf.Session = tf.Session(graph=g)
        self.session.run(self.init_op)

    def dim(self) -> int:
        return 512

    def embed_corpus(self, corpus: Corpus, output_path: str) -> np.ndarray:
        rows: int = len(corpus)
        cols: int = corpus.sentences_per_sample
        texts = corpus.raw_texts()
        embeddings = self.run_use(texts)
        result = np.ndarray((rows, cols, self.dim()), dtype=np.float32)
        for idx, vector in enumerate(embeddings):
            row_idx = int(np.floor(idx / cols))
            col_idx = 0 if cols == 1 else row_idx % cols
            result[row_idx, col_idx, :] = vector
        if output_path is not None:
            np.save(output_path, result)
        return result

    def batcher(self, params, batch: List[List[str]]) -> np.ndarray:
        batch = [" ".join(sent) if sent != [] else ['.'] for sent in batch]
        embeddings = self.run_use(batch)
        embeddings = np.vstack(embeddings)
        return embeddings

    def _init_graph(self):
        g = tf.Graph()
        with g.as_default():
            text_input = tf.placeholder(dtype=tf.string, shape=[None])
            embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-xling-many/1")
            embedded_text = embed(text_input)
            init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])
        g.finalize()
        return g, init_op, text_input, embedded_text

    def run_use(self, texts: List[str]) -> List[np.ndarray]:
        res: List[np.ndarray] = []
        for text in texts:
            embedding = self.session.run(self.embed_op, feed_dict={self.embed_input: [text]})
            res.append(embedding[0])
        return res
