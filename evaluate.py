import logging
import os
from pathlib import Path

import fire
import json

from analyzer import PolishAnalyzer
from methods.base import EmbeddingBase
from methods.utils import cached
from sentevalpl.engine import SE

root_dir = os.path.dirname(os.path.realpath(__file__))


class SentEvaluator(object):

    def random(self, **kwargs):
        from methods.word_vectors import RandomEmbedding
        method = RandomEmbedding()
        self.evaluate(method, "random", **kwargs)

    def word2vec(self, **kwargs):
        from methods.word_vectors import KeyedVectorsEmbedding
        method = KeyedVectorsEmbedding(Path(root_dir, "resources/word2vec/word2vec_100_3_polish.bin"))
        self.evaluate(method, "word2vec", **kwargs)

    def glove(self, **kwargs):
        from methods.word_vectors import KeyedVectorsEmbedding
        method = KeyedVectorsEmbedding(Path(root_dir, "resources/glove/glove_100_3_polish.txt"))
        self.evaluate(method, "glove", **kwargs)

    def fasttext(self, **kwargs):
        from methods.word_vectors import KeyedVectorsEmbedding
        method = KeyedVectorsEmbedding(Path(root_dir, "resources/fasttext/fasttext_100_3_polish.bin"))
        self.evaluate(method, "fasttext", **kwargs)

    def elmo_all(self, **kwargs):
        from methods.word_vectors import ElmoEmbedding
        method = ElmoEmbedding(Path(root_dir, "resources/elmo/"), layers="all")
        self.evaluate(method, "elmo_all", **kwargs)

    def elmo_top(self, **kwargs):
        from methods.word_vectors import ElmoEmbedding
        method = ElmoEmbedding(Path(root_dir, "resources/elmo/"), layers="top")
        self.evaluate(method, "elmo_top", **kwargs)

    def flair(self, **kwargs):
        from methods.word_vectors import FlairEmbedding
        method = FlairEmbedding()
        self.evaluate(method, "flair", **kwargs)

    def bert(self, **kwargs):
        from methods.word_vectors import BertEmbedding
        method = BertEmbedding()
        self.evaluate(method, "bert", **kwargs)

    def laser(self, **kwargs):
        from methods.laser import LaserEmbedding
        method = LaserEmbedding()
        self.evaluate(method, "laser", **kwargs)

    def use(self, **kwargs):
        from methods.use import USEEmbedding
        method = USEEmbedding()
        self.evaluate(method, "use", **kwargs)

    def evaluate(self, method: EmbeddingBase, method_name: str, **kwargs):
        logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)
        logging.root.setLevel(logging.DEBUG)
        params = {
            "task_path": os.path.join(root_dir, "resources"),
            "usepytorch": True,
            "kfold": 5,
            "lemmatize": True,
            "batch_size": 512,
            "classifier": {"nhid": 100, "optim": "rmsprop", "batch_size": 128, "tenacity": 3, "epoch_size": 10},
            "analyzer": PolishAnalyzer()
        }
        params.update(kwargs)
        cache_dir = Path(root_dir, f".cache/{method_name}")
        cache_dir.mkdir(parents=True, exist_ok=True)
        se = SE(params, cached(method.batcher, cache_dir), method.prepare)
        transfer_tasks = ["WCCRS_HOTELS", "WCCRS_MEDICINE", "SICKEntailment", "SICKRelatedness"]
        results = se.eval(transfer_tasks)
        del results["SICKRelatedness"]["yhat"]
        results = {"method": method_name, "results": results}
        logging.info(results)
        with open(os.path.join(root_dir, "results.txt"), "a+") as output_file:
            output_file.write(json.dumps(results))
            output_file.write("\n")


if __name__ == '__main__':
    fire.Fire(SentEvaluator)