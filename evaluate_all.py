import logging
import subprocess
import os

logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)
logging.root.setLevel(logging.DEBUG)


def evaluate(name: str, params=None):
    if params is None: params = {}
    cmd = ["python", "evaluate.py", name]
    for key, val in params.items():
        cmd.append(f"--{key}")
        cmd.append(str(val))
    logging.info("running %s", cmd.__repr__())
    subprocess.run(cmd)


if __name__ == '__main__':
    root_dir = os.path.dirname(os.path.realpath(__file__))
    results = os.path.join(root_dir, "results.txt")
    if os.path.exists(results): os.remove(results)
    evaluate("random")
    evaluate("word2vec")
    evaluate("glove")
    evaluate("fasttext")
    evaluate("elmo_all")
    evaluate("elmo_top")
    evaluate("flair", {"batch-size": 256})
    evaluate("bert", {"batch-size": 32})
    evaluate("laser", {"batch-size": 10000})
    evaluate("use")