import logging
import os
from collections import namedtuple
from pathlib import Path

import fire
import json

from typing import Union

from sentevalpl.tasks import get_task_names
from utils.analyzer import PolishAnalyzer
from methods.base import EmbeddingBase
from methods.utils import cached
from sentevalpl.engine import SE

root_dir = os.path.dirname(os.path.realpath(__file__))


TaskDef = namedtuple("TaskDef", ("id", "name", ))

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
        from methods.huggingface import HuggingfaceModelEmbedding
        method = HuggingfaceModelEmbedding("bert-base-multilingual-cased", layers="top")
        self.evaluate(method, "bert", **kwargs)

    def roberta_base_top(self, **kwargs):
        from methods.huggingface import HuggingfaceModelEmbedding
        method = HuggingfaceModelEmbedding("resources/roberta_base_transformers/", layers="top")
        self.evaluate(method, "roberta_base_top", **kwargs)

    def roberta_base_all(self, **kwargs):
        from methods.huggingface import HuggingfaceModelEmbedding
        method = HuggingfaceModelEmbedding("resources/roberta_base_transformers/", layers="all")
        self.evaluate(method, "roberta_base_all", **kwargs)

    def roberta_large_top(self, **kwargs):
        from methods.huggingface import HuggingfaceModelEmbedding
        method = HuggingfaceModelEmbedding("resources/roberta_large_transformers/", layers="top")
        self.evaluate(method, "roberta_large_top", **kwargs)

    def roberta_large_all(self, **kwargs):
        from methods.huggingface import HuggingfaceModelEmbedding
        method = HuggingfaceModelEmbedding("resources/roberta_large_transformers/", layers="all")
        self.evaluate(method, "roberta_large_all", **kwargs)

    def xlmr_base_top(self, **kwargs):
        from methods.huggingface import HuggingfaceModelEmbedding
        method = HuggingfaceModelEmbedding("xlm-roberta-base", layers="top")
        self.evaluate(method, "xlmr_base_top", **kwargs)

    def xlmr_base_all(self, **kwargs):
        from methods.huggingface import HuggingfaceModelEmbedding
        method = HuggingfaceModelEmbedding("xlm-roberta-base", layers="all")
        self.evaluate(method, "xlmr_base_all", **kwargs)

    def xlmr_large_top(self, **kwargs):
        from methods.huggingface import HuggingfaceModelEmbedding
        method = HuggingfaceModelEmbedding("xlm-roberta-large", layers="top")
        self.evaluate(method, "xlmr_large_top", **kwargs)

    def xlmr_large_all(self, **kwargs):
        from methods.huggingface import HuggingfaceModelEmbedding
        method = HuggingfaceModelEmbedding("xlm-roberta-large", layers="all")
        self.evaluate(method, "xlmr_large_all", **kwargs)

    def huggingface(self, **kwargs):
        model_name = kwargs.get("model_name")
        from methods.huggingface import HuggingfaceModelEmbedding
        method = HuggingfaceModelEmbedding(model_name, layers="top")
        self.evaluate(method, "huggingface_" + model_name.replace("/", "_"), **kwargs)

    def laser(self, **kwargs):
        from methods.laser import LaserEmbedding
        method = LaserEmbedding()
        self.evaluate(method, "laser", **kwargs)

    def use(self, **kwargs):
        from methods.use import USEEmbedding
        method = USEEmbedding()
        self.evaluate(method, "use", **kwargs)

    def labse(self, **kwargs):
        self.sentence_transformers(model_name="sentence-transformers/LaBSE", **kwargs)

    def sentence_transformers(self, **kwargs):
        from methods.sentence_transformer import SentenceTransformersEmbedding
        model_name = kwargs.get("model_name")
        method = SentenceTransformersEmbedding(model_name)
        self.evaluate(method, "sentence_transformers_" + model_name.replace("/", "_"), **kwargs)

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
        transfer_tasks = get_task_names()
        results = se.eval(transfer_tasks)
        for key, val in results.items():
            if "yhat" in val.keys():
                del val["yhat"]
        results = {"method": method_name, "results": results}
        logging.info(results)
        with open(os.path.join(root_dir, "results.txt"), "a+") as output_file:
            output_file.write(json.dumps(results))
            output_file.write("\n")


if __name__ == '__main__':
    fire.Fire(SentEvaluator)