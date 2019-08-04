import logging
import os
import numpy as np
from senteval.tools.validation import SplitClassifier


class SentEvalClassifier(object):

    def __init__(self, task_path: str, task_name: str, nclasses: int, seed=1111):
        self.seed = seed
        self.nclasses = nclasses
        self.task_name = task_name
        logging.debug("***** Transfer task : %s classification *****\n\n", self.task_name)

        train = self.load_file(os.path.join(task_path, "train.txt"))
        dev = self.load_file(os.path.join(task_path, "dev.txt"))
        test = self.load_file(os.path.join(task_path, "test.txt"))
        self.data = {"train": train, "dev": dev, "test": test}

    def do_prepare(self, params, prepare):
        samples = self.data["train"]["X"] + self.data["dev"]["X"] + self.data["test"]["X"]
        return prepare(params, samples)

    def load_file(self, fpath):
        results = {"X": [], "y": []}
        with open(fpath, "r", encoding="utf-8") as input_file:
            for line in input_file:
                sample = line.strip().split(' ', 1)
                results["y"].append(int(sample[0]))
                results["X"].append(sample[1].split())
        assert max(results["y"]) == self.nclasses - 1
        return results

    def run(self, params, batcher):
        data = {"train": {}, "dev": {}, "test": {}}
        bsize = params.batch_size

        for key in self.data:
            params.batcher_dataset = key
            logging.info("Computing embedding for {0}".format(key))
            # Sort to reduce padding
            sorted_data = sorted(zip(self.data[key]["X"], self.data[key]["y"]), key=lambda z: (len(z[0]), z[1]))
            self.data[key]["X"], self.data[key]["y"] = map(list, zip(*sorted_data))

            data[key]["X"] = []
            for ii in range(0, len(self.data[key]["y"]), bsize):
                params.batcher_offset = str(ii)
                batch = self.data[key]["X"][ii:ii + bsize]
                embeddings = batcher(params, batch)
                data[key]["X"].append(embeddings)
            data[key]["X"] = np.vstack(data[key]["X"])
            data[key]["y"] = np.array(self.data[key]["y"])
            logging.info("Computed {0} embeddings".format(key))

        config_classifier = {
            "nclasses": self.nclasses,
            "seed": self.seed,
            "usepytorch": params.usepytorch,
            "classifier": params.classifier
        }

        clf = SplitClassifier(
            X={"train": data["train"]["X"],"valid": data["dev"]["X"],"test": data["test"]["X"]},
            y={"train": data["train"]["y"],"valid": data["dev"]["y"],"test": data["test"]["y"]},
            config=config_classifier
        )

        devacc, testacc = clf.run()
        logging.debug("\nDev acc : {0} Test acc : {1} for {2} classification\n".format(devacc, testacc, self.task_name))
        return {"devacc": devacc, "acc": testacc, "ndev": len(data["dev"]["X"]), "ntest": len(data["test"]["X"])}
