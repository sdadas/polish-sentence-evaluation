import os
import numpy as np
from typing import List
from torch import Tensor, hub

from methods.base import EmbeddingBase
from fairseq.models.roberta import RobertaModel, RobertaHubInterface
from fairseq import hub_utils

class RobertaEmbedding(EmbeddingBase):

    def __init__(self, path: str, bpe: str="sentencepiece", bpe_filename:str="sentencepiece.model"):
        self.model: RobertaHubInterface = self._load_model(path, bpe, bpe_filename)
        self.model.eval()
        self.size = self.model.model.args.encoder_embed_dim

    def _load_model(self, path: str, bpe: str, bpe_filename:str) -> RobertaHubInterface:
        if path == "xlmr.large":
            return hub.load("pytorch/fairseq", path, force_reload=True)
        else:
            checkpoint_file = "model.pt" if os.path.exists(os.path.join(path, "model.pt")) else "checkpoint_best.pt"
            loaded = hub_utils.from_pretrained(
                model_name_or_path=path,
                checkpoint_file=checkpoint_file,
                data_name_or_path=path,
                bpe=bpe,
                sentencepiece_vocab=os.path.join(path, bpe_filename),
                load_checkpoint_heads=True,
                archive_map=RobertaModel.hub_models(),
                cpu=True
            )
            return RobertaHubInterface(loaded['args'], loaded['task'], loaded['models'][0])

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
