import logging
import os
from pathlib import Path

from analyzer import PolishAnalyzer
from methods.laser import LaserEmbedding
from methods.word_vectors import RandomEmbedding, KeyedVectorsEmbedding, ElmoEmbedding, FlairEmbedding
from sentevalpl.engine import SE

root_dir = os.path.dirname(os.path.realpath(__file__))
params = {"task_path": os.path.join(root_dir, "resources"), "usepytorch": True, "kfold": 5, "lemmatize": True, "batch_size": 512}
params["classifier"] = {"nhid": 100, "optim": "rmsprop", "batch_size": 128, "tenacity": 3, "epoch_size": 10}
params["analyzer"] = PolishAnalyzer()
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)
logging.root.setLevel(logging.DEBUG)

if __name__ == "__main__":
    #method = KeyedVectorsEmbedding(Path(root_dir, "resources/word2vec/word2vec_100_3_polish.bin"))
    #method = RandomEmbedding()
    #method = ElmoEmbedding(Path(root_dir, "resources/elmo"))
    #method = FlairEmbedding()
    method = LaserEmbedding()
    #method = USEEmbedding()
    se = SE(params, method.batcher, method.prepare)
    transfer_tasks = ["WCCRS_HOTELS", "WCCRS_MEDICINE", "SICKEntailment", "SICKRelatedness"]
    results = se.eval(transfer_tasks)
    print(results)