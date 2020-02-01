import os
import numpy as np
from typing import List
from torch import Tensor

from methods.base import EmbeddingBase
from fairseq.models.roberta import RobertaModel, RobertaHubInterface
from fairseq import hub_utils

class RobertaEmbedding(EmbeddingBase):

    def __init__(self, path: str, bpe: str="sentencepiece", bpe_filename:str="sentencepiece.model"):
        loaded = hub_utils.from_pretrained(
            model_name_or_path=path,
            checkpoint_file="checkpoint_best.pt",
            data_name_or_path=path,
            bpe=bpe,
            sentencepiece_vocab=os.path.join(path, bpe_filename),
            load_checkpoint_heads=True,
            archive_map=RobertaModel.hub_models(),
            cpu=True
        )
        self.model: RobertaHubInterface = RobertaHubInterface(loaded['args'], loaded['task'], loaded['models'][0])
        self.model.eval()
        self.size = 768

    def dim(self) -> int:
        return self.size

    def _get_vector(self, sentence: Tensor) -> np.ndarray:
        emb: np.ndarray = sentence.detach().numpy()
        res = np.zeros(self.size, dtype=np.float32)
        for idx in range(emb.shape[1]):
            vec = emb[0][idx][:]
            vec = vec / np.linalg.norm(vec, ord=2)
            res += vec
        res /= emb.shape[1]
        return res

    def batcher(self, params, batch: List[List[str]]) -> np.ndarray:
        batch = [" ".join(sent) if sent != [] else "." for sent in batch]
        embeddings = []
        for sent in batch:
            tokens = self.model.encode(sent)
            embedding: Tensor = self.model.extract_features(tokens)
            embeddings.append(self._get_vector(embedding))
        embeddings = np.vstack(embeddings)
        return embeddings
