import logging
import os
from pathlib import Path

import fire
import json

from typing import Union

from utils.analyzer import PolishAnalyzer
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
        path: Path = Path(root_dir, "resources/word2vec/word2vec_100_3_polish.bin")
        self.evaluate_keyed_vectors(path, "word2vec", **kwargs)

    def glove(self, **kwargs):
        path: Path = Path(root_dir, "resources/glove/glove_100_3_polish.txt")
        self.evaluate_keyed_vectors(path, "glove", **kwargs)

    def fasttext(self, **kwargs):
        path: Path = Path(root_dir, "resources/fasttext/fasttext_100_3_polish.bin")
        self.evaluate_keyed_vectors(path, "fasttext", **kwargs)

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

    def roberta(self, **kwargs):
        from methods.roberta import RobertaEmbedding
        method = RobertaEmbedding("resources/roberta/")
        self.evaluate(method, "roberta", **kwargs)

    def xlmr(self, **kwargs):
        from methods.roberta import RobertaEmbedding
        method = RobertaEmbedding("resources/xlmr.large/", bpe="sentencepiece", bpe_filename="sentencepiece.bpe.model")
        self.evaluate(method, "xlmr", **kwargs)

    def laser(self, **kwargs):
        from methods.laser import LaserEmbedding
        method = LaserEmbedding()
        self.evaluate(method, "laser", **kwargs)

    def use(self, **kwargs):
        from methods.use import USEEmbedding
        method = USEEmbedding()
        self.evaluate(method, "use", **kwargs)

    def evaluate_keyed_vectors(self, path: Union[Path, str], name: str, **kwargs):
        if isinstance(path, str): path = Path(path)
        if kwargs.get("sif"):
            from methods.sif import SIFEmbedding
            name = name + "_sif"
            method = SIFEmbedding(path)
        else:
            from methods.word_vectors import KeyedVectorsEmbedding
            pooling = "avg"
            if kwargs.get("pooling") and kwargs.get("pooling") != "avg":
                pooling = kwargs.get("pooling")
                name = name + "_" +pooling
            method = KeyedVectorsEmbedding(path, pooling=pooling)
        self.evaluate(method, name, **kwargs)

    def evaluate(self, method: EmbeddingBase, method_name: str, **kwargs):
        logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)
        logging.root.setLevel(logging.DEBUG)
        params = {
            "task_path": os.path.join(root_dir, "resources"),
            "usepytorch": True,
            "kfold": 5,
            "lemmatize": True,
            "batch_size": 512,
            "classifier": {"nhid": 50, "optim": "rmsprop", "batch_size": 128, "tenacity": 3, "epoch_size": 10},
            "analyzer": PolishAnalyzer()
        }
        params.update(kwargs)
        cache_dir = Path(root_dir, f".cache/{method_name}")
        cache_dir.mkdir(parents=True, exist_ok=True)
        se = SE(params, cached(method.batcher, cache_dir), method.prepare)
        transfer_tasks = ["WCCRS_HOTELS", "WCCRS_MEDICINE", "SICKEntailment", "SICKRelatedness", "8TAGS"]
        results = se.eval(transfer_tasks)
        del results["SICKRelatedness"]["yhat"]
        results = {"method": method_name, "results": results}
        logging.info(results)
        with open(os.path.join(root_dir, "results.txt"), "a+") as output_file:
            output_file.write(json.dumps(results))
            output_file.write("\n")


if __name__ == '__main__':
    fire.Fire(SentEvaluator)