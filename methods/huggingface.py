from typing import List

import numpy as np
import torch.cuda
import torch.nn.functional as F
from transformers import AutoConfig, AutoTokenizer, AutoModel
from methods.base import EmbeddingBase


class HuggingfaceModelEmbedding(EmbeddingBase):

    def __init__(self, model_name_or_path: str, layers:str="top", mini_batch_size: int=32):
        assert layers in ("top", "all")
        self.config = AutoConfig.from_pretrained(model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        self.model = AutoModel.from_pretrained(model_name_or_path)
        self.dev = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.layers = layers
        self.mini_batch_size = mini_batch_size
        if torch.cuda.is_available():
            self.model.half()
        self.model.eval()
        self.model.to(self.dev)

    def dim(self) -> int:
        return self.config.hidden_size

    def get_hidden_states_for_batch(self, texts: List[str]):
        tokens = self.tokenizer(texts, max_length=512, truncation=True, padding=True, return_tensors="pt").to(self.dev)
        output = self.model(**tokens, output_hidden_states=True)
        if self.layers == "top":
            hidden_states = output.hidden_states[-1]
            return self.get_hidden_states_for_layer(hidden_states)
        elif self.layers == "all":
            results = [self.get_hidden_states_for_layer(hs) for hs in output.hidden_states]
            return np.hstack(results)

    def get_hidden_states_for_layer(self, hidden_states):
        num_words = hidden_states.size()[1]
        hidden_states = F.normalize(hidden_states, p=2, dim=2)
        hidden_states = torch.sum(hidden_states, 1)
        return (hidden_states / num_words).detach().cpu().numpy()

    def batcher(self, params, batch: List[List[str]]) -> np.ndarray:
        batch = [" ".join(sent) if sent != [] else ['.'] for sent in batch]
        results = []
        for i in range(0, len(batch), self.mini_batch_size):
            texts = batch[i:i + self.mini_batch_size]
            results.append(self.get_hidden_states_for_batch(texts))
        return np.vstack(results)
