import logging
import subprocess
import os
import json
import functools

from sentevalpl.tasks import get_task_names
from utils.table import TablePrinter, TableColumn

logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)
logging.root.setLevel(logging.DEBUG)


def evaluate(name: str, params=None):
    if params is None: params = {}
    cmd = ["python3", "evaluate.py", name]
    for key, val in params.items():
        cmd.append(f"--{key}")
        cmd.append(str(val))
    logging.info("running %s", cmd.__repr__())
    subprocess.run(cmd)

def print_results(results_file: str):
    header = ["method"]
    header.append("average")
    header.extend(get_task_names())
    table = [header]
    score = lambda val, ds: "%.2f" % (val["results"][ds].get("acc", val["results"][ds].get("spearman", 0) * 100),)
    with open(results_file, "r", encoding="utf-8") as input_file:
        for line in input_file:
            obj = json.loads(line)
            s = functools.partial(score, obj)
            row = [obj["method"]]
            row_scores = [s(task_name) for task_name in get_task_names()]
            avg_score = sum(float(val) for val in row_scores) / len(row_scores)
            row.append("%.2f" % (avg_score,))
            row.extend(row_scores)
            print(" & ".join(row))
            table.append(row)
    printer: TablePrinter = TablePrinter()
    for idx in range(1, len(header)): printer.column(idx, align=TableColumn.ALIGN_CENTER, width=15)
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
    evaluate("bert", {"batch-size": 256})
    evaluate("roberta_base_top", {"batch-size": 256})
    evaluate("roberta_base_all", {"batch-size": 256})
    evaluate("roberta_large_top", {"batch-size": 256})
    evaluate("roberta_large_all", {"batch-size": 256})
    evaluate("xlmr_base_top", {"batch-size": 256})
    evaluate("xlmr_base_all", {"batch-size": 256})
    evaluate("xlmr_large_top", {"batch-size": 256})
    evaluate("xlmr_large_all", {"batch-size": 256})
    evaluate("flair", {"batch-size": 256})
    evaluate("laser", {"batch-size": 10000})
    evaluate("use")
    evaluate("labse")
    evaluate("huggingface", {"model_name": "allegro/herbert-base-cased"})
    evaluate("huggingface", {"model_name": "allegro/herbert-large-cased"})
    evaluate("sentence_transformers", {"model_name": "distiluse-base-multilingual-cased-v2"})
    evaluate("sentence_transformers", {"model_name": "xlm-r-distilroberta-base-paraphrase-v1"})
    evaluate("sentence_transformers", {"model_name": "xlm-r-bert-base-nli-stsb-mean-tokens"})
    evaluate("sentence_transformers", {"model_name": "distilbert-multilingual-nli-stsb-quora-ranking"})
    evaluate("sentence_transformers", {"model_name": "paraphrase-multilingual-mpnet-base-v2"})
    evaluate("sentence_transformers", {"model_name": "paraphrase-multilingual-MiniLM-L12-v2"})

    print_results(results)