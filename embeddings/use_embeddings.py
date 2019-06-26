from pathlib import Path
from typing import List

from data import Corpus, Sent
from embeddings.base import EmbeddingBase
import tensorflow_hub as hub
import tensorflow as tf
from tf_sentencepiece import sentencepiece_processor_ops
import numpy as np

class USEEmbedding(EmbeddingBase):

    def embed(self, sentence: Sent) -> np.ndarray:
        res = self.run_use(sentence.raw)
        return res[0]

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

    def run_use(self, texts: List[str]) -> List[np.ndarray]:
        g = tf.Graph()
        with g.as_default():
            text_input = tf.placeholder(dtype=tf.string, shape=[None])
            embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-xling-many/1")
            embedded_text = embed(text_input)
            init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])
        g.finalize()

        session = tf.Session(graph=g)
        session.run(init_op)
        res: List[np.ndarray] = []
        for text in texts:
            embedding = session.run(embedded_text, feed_dict={text_input: [text]})
            res.append(embedding[0])
        return res
