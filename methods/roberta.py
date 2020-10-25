import os
import numpy as np
from typing import List
from torch import Tensor, hub

from methods.base import EmbeddingBase
from fairseq.models.roberta import RobertaModel, RobertaHubInterface
from fairseq import hub_utils


class RobertaEmbedding(EmbeddingBase):

    def __init__(self, path: str, layers:str="top", bpe: str="sentencepiece", bpe_filename:str="sentencepiece.model"):
        assert layers == "all" or layers == "top"
        self.layers = layers
        self.model: RobertaHubInterface = self._load_model(path, bpe, bpe_filename)
        self.model.eval()
        self.model.cuda()
        self.size = self.model.model.args.encoder_embed_dim
        if layers == "all": self.size *= (self.model.model.args.encoder_layers + 1)

    def _load_model(self, path: str, bpe: str, bpe_filename:str) -> RobertaHubInterface:
        if path == "xlmr.large" or path == "xlmr.base":
            return hub.load("pytorch/fairseq", path, force_reload=True)
        else:
            checkpoint_file = "model.pt" if os.path.exists(os.path.join(path, "model.pt")) else "checkpoint_best.pt"
            loaded = hub_utils.from_pretrained(
                model_name_or_path=path,
                checkpoint_file=checkpoint_file,
                data_name_or_path=path,
                bpe=bpe,
                sentencepiece_vocab=os.path.join(path, bpe_filename),
                sentencepiece_model=os.path.join(path, bpe_filename),
                load_checkpoint_heads=True,
                archive_map=RobertaModel.hub_models(),
                cpu=False
            )
            return RobertaHubInterface(loaded['args'], loaded['task'], loaded['models'][0])

    def dim(self) -> int:
        return self.size

    def _get_vector_top(self, sentence: Tensor) -> np.ndarray:
        emb: np.ndarray = sentence.detach().cpu().numpy()
        res = np.zeros(self.size, dtype=np.float32)
        for idx in range(emb.shape[1]):
            vec = emb[0][idx][:]
            vec = vec / np.linalg.norm(vec, ord=2)
            res += vec
        res /= emb.shape[1]
        return res

    def _get_vector_all(self, sentence) -> np.ndarray:
        emb: List = [layer.detach().cpu().numpy() for layer in sentence]
        res = np.zeros(self.size, dtype=np.float32)
        num_words = emb[0].shape[1]
        for idx in range(num_words):
            word_res = []
            for layer_emb in emb:
                word_res.append(layer_emb[0][idx][:])
            vec = np.hstack(word_res)
            vec = vec / np.linalg.norm(vec, ord=2)
            res += vec
        res /= num_words
        return res

    def batcher(self, params, batch: List[List[str]]) -> np.ndarray:
        batch = [" ".join(sent) if sent != [] else "." for sent in batch]
        embeddings = []
        all_layers = self.layers == "all"
        for sent in batch:
            tokens = self.model.encode(sent)
            embedding = self.model.extract_features(tokens, return_all_hiddens=all_layers)
            embeddings.append(self._get_vector_all(embedding) if all_layers else self._get_vector_top(embedding))
        embeddings = np.vstack(embeddings)
        return embeddings
