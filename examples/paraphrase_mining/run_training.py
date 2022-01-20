import logging
import json
import os.path
from dataclasses import dataclass, field
from typing import List, Any

from sentence_transformers import models, losses, SentenceTransformer, InputExample
from sentence_transformers.datasets import NoDuplicatesDataLoader
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from tqdm import tqdm
from transformers import HfArgumentParser

from extensions.LSTMPooling import LSTMPooling

logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)
logging.root.setLevel(logging.DEBUG)


@dataclass
class MNRParaphraseArgs:
    model: str = field(
        metadata={"help": "The name or path to the model"},
    )
    input_paths: str = field(
        metadata={"help": "Comma separated list of input JSONL files"},
    )
    output_path: str = field(
        default="./output",
        metadata={"help": "Directory to store fine-tuned model"},
    )
    eval_path: str = field(
        default=None,
        metadata={"help": "Path to the evaluation JSONL file"},
    )
    eval_steps: int = field(
        default=10_000,
        metadata={"help": "Number of steps after which the model is evaluated"},
    )
    fp16: bool = field(
        default=True,
        metadata={"help": "Use AMP for mixed precision training"},
    )
    batch_size: int = field(
        default=32,
        metadata={"help": "Batch size"},
    )
    lr: float = field(
        default=2e-6,
        metadata={"help": "Learning rate"},
    )
    num_train_epochs: int = field(
        default=3,
        metadata={"help": "Number of training epochs"},
    )
    warmup_ratio: float = field(
        default=0.1,
        metadata={"help": "Ratio of warmup steps"},
    )
    max_levenshtein: float = field(
        default=0.99,
        metadata={"help": "Discard all examples with Levenshtein similarity higher than this value"},
    )
    pooling_type: str = field(
        default="lstm",
        metadata={"help": "Type of the pooling layer to use"},
    )
    pooler_output_dim: int = field(
        default=2048,
        metadata={"help": "Output representation size for pooler (applies only to some pooler types)"},
    )


class MNRParaphraseTrainer:

    def __init__(self, args: MNRParaphraseArgs):
        self.args = args
        self.base_model = models.Transformer(self.args.model)
        self.pooler = self._create_pooler()

    def train(self):
        model = SentenceTransformer(modules=[self.base_model, self.pooler])
        loss = losses.MultipleNegativesRankingLoss(model)
        loader: Any = self._load_data()
        evaluator = self._load_evaluator()
        warmup_steps = int(len(loader) * self.args.num_train_epochs * self.args.warmup_ratio)
        logging.info("Beginning training")
        model.fit(
            train_objectives=[(loader, loss)],
            epochs=self.args.num_train_epochs,
            warmup_steps=warmup_steps,
            output_path=self.args.output_path,
            show_progress_bar=False,
            use_amp=self.args.fp16,
            evaluation_steps=self.args.eval_steps if self.args.eval_path is not None else 0,
            evaluator=evaluator,
            checkpoint_path=self.args.output_path,
            checkpoint_save_steps=self.args.eval_steps,
            checkpoint_save_total_limit=5,
            optimizer_params={'lr': self.args.lr}
        )

    def _load_data(self):
        input_paths = [input_path.strip() for input_path in self.args.input_paths.split(",")]
        samples = []
        for input_path in input_paths:
            self._load_file(input_path, samples)
        logging.info("Loaded %d examples", len(samples))
        return NoDuplicatesDataLoader(samples, self.args.batch_size)

    def _load_evaluator(self):
        if self.args.eval_path is None:
            return None
        sentences1, sentences2, scores = [], [], []
        with open(self.args.eval_path, "r", encoding="utf-8") as input_file:
            for line in input_file:
                value = json.loads(line.strip())
                sent1, sent2, label = value["sent1"], value["sent2"], value["score"]["similarity"]
                sentences1.append(sent1)
                sentences2.append(sent2)
                scores.append(label)
        dataset_name = os.path.basename(self.args.eval_path)
        return EmbeddingSimilarityEvaluator(sentences1, sentences2, scores, write_csv=False, name=dataset_name)

    def _load_file(self, input_path: str, samples: List[InputExample]):
        logging.info("Loading examples from %s", input_path)
        with open(input_path, "r", encoding="utf-8") as input_file:
            for line in tqdm(input_file):
                value = json.loads(line.strip())
                levenshtein = value["score"]["levenshtein"]
                if levenshtein > self.args.max_levenshtein:
                    continue
                sent1, sent2 = value["sent1"], value["sent2"]
                label = value["score"]["label"]
                if label is not None and label != "ENTAILMENT":
                    continue
                samples.append(InputExample(texts=[sent1, sent2]))

    def _create_pooler(self):
        pooler = self.args.pooling_type
        emb_dim = self.base_model.get_word_embedding_dimension()
        if pooler in ("mean", "max", "cls"):
            return models.Pooling(emb_dim, pooling_mode=pooler)
        elif pooler == "lstm":
            return LSTMPooling(emb_dim, self.args.pooler_output_dim)
        elif pooler == "bilstm":
            return LSTMPooling(emb_dim, int(self.args.pooler_output_dim / 2), bidirectional=True)
        else:
            raise ValueError(f"Unsupported pooler {pooler}")


if __name__ == '__main__':
    parser = HfArgumentParser([MNRParaphraseArgs])
    args = parser.parse_args_into_dataclasses()[0]
    trainer = MNRParaphraseTrainer(args)
    trainer.train()
