import torch
from torch import Tensor
from torch import nn
from typing import Dict
import os
import json


class LSTMPooling(nn.Module):

    def __init__(self, word_embedding_dimension, hidden_size, bidirectional=False):
        super(LSTMPooling, self).__init__()
        self.config_keys = ['word_embedding_dimension', 'hidden_size', 'bidirectional']
        self.word_embedding_dimension = word_embedding_dimension
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(word_embedding_dimension, hidden_size, batch_first=True, bidirectional=bidirectional)

    def forward(self, features: Dict[str, Tensor]):
        token_embeddings = features['token_embeddings']
        attention_mask = features['attention_mask']
        sentence_lengths = torch.sum(attention_mask, dim=1, dtype=torch.int64).cpu()
        batch_size = sentence_lengths.size()[0]

        packed = nn.utils.rnn.pack_padded_sequence(token_embeddings, sentence_lengths, batch_first=True, enforce_sorted=False)
        packed = self.lstm(packed)
        unpack = nn.utils.rnn.pad_packed_sequence(packed[0], batch_first=True)
        output_states = unpack[0][torch.arange(batch_size), (unpack[1] - 1)]
        features.update({'sentence_embedding': output_states})
        return features

    def __repr__(self):
        return "LSTMPooling({})".format(self.get_config_dict())

    def get_word_embedding_dimension(self):
        return self.hidden_size * (2 if self.bidirectional else 1)

    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}

    def save(self, output_path):
        with open(os.path.join(output_path, 'config.json'), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

        torch.save(self.state_dict(), os.path.join(output_path, 'pytorch_model.bin'))

    @staticmethod
    def load(input_path):
        with open(os.path.join(input_path, 'config.json')) as fIn:
            config = json.load(fIn)

        model = LSTMPooling(**config)
        model.load_state_dict(torch.load(os.path.join(input_path, 'pytorch_model.bin'), map_location=torch.device('cpu')))
        return model
