import logging
import subprocess
import os
import json
import functools

from utils.table import TablePrinter, TableColumn

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

def print_results(results_file: str):
    header = ["method", "WCCRS Hotels", "WCCRS Medicine", "SICK-E", "SICK-R", "8TAGS"]
    table = [header]
    score = lambda val, ds: "%.2f" % (val["results"][ds].get("acc", val["results"][ds].get("pearson", 0) * 100),)
    with open(results_file, "r", encoding="utf-8") as input_file:
        for line in input_file:
            obj = json.loads(line)
            s = functools.partial(score, obj)
            row = [obj["method"], s("WCCRS_HOTELS"), s("WCCRS_MEDICINE"), s("SICKEntailment"), s("SICKRelatedness"), s('8TAGS')]
            table.append(row)
    printer: TablePrinter = TablePrinter()
    for idx in range(1, 6): printer.column(idx, align=TableColumn.ALIGN_CENTER, width=15)
    printer.print(table)


if __name__ == '__main__':
    root_dir = os.path.dirname(os.path.realpath(__file__))
    results = os.path.join(root_dir, "results.txt")
    if os.path.exists(results): os.remove(results)
    evaluate("random")
    evaluate("word2vec")
    evaluate("word2vec", {"sif": "true"})
    evaluate("word2vec", {"pooling": "concat"})
    evaluate("glove")
    evaluate("glove", {"sif": "true"})
    evaluate("glove", {"pooling": "concat"})
    evaluate("fasttext")
    evaluate("fasttext", {"sif": "true"})
    evaluate("fasttext", {"pooling": "concat"})
    evaluate("elmo_all")
    evaluate("elmo_top")
    evaluate("bert", {"batch-size": 32})
    evaluate("roberta", {"batch-size": 256})
    evaluate("flair", {"batch-size": 256})
    evaluate("laser", {"batch-size": 10000})
    evaluate("use")
    print_results(results)