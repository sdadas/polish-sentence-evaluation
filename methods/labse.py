import logging
from typing import List

import bert
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from methods.base import EmbeddingBase

logging.root.setLevel(logging.DEBUG)


class LABSEEmbedding(EmbeddingBase):

    def __init__(self):
        self.max_seq_length = 128
        self.labse_model = None
        self.labse_layer = None
        self.tokenizer = None


    def _init_model(self):
        labse_model, labse_layer = self._create_model()
        self.labse_model = labse_model
        self.labse_layer = labse_layer
        self.tokenizer = self._create_tokenizer()

    def dim(self) -> int:
        return 768

    def _create_model(self):
        model_url = "https://tfhub.dev/google/LaBSE/1"
        labse_layer = hub.KerasLayer(model_url, trainable=False)
        input_word_ids = tf.keras.layers.Input(shape=(self.max_seq_length,), dtype=tf.int32, name="input_word_ids")
        input_mask = tf.keras.layers.Input(shape=(self.max_seq_length,), dtype=tf.int32, name="input_mask")
        segment_ids = tf.keras.layers.Input(shape=(self.max_seq_length,), dtype=tf.int32, name="segment_ids")
        pooled_output, _ = labse_layer([input_word_ids, input_mask, segment_ids])
        pooled_output = tf.keras.layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=1))(pooled_output)
        labse_model = tf.keras.Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=pooled_output)
        return labse_model, labse_layer

    def _create_tokenizer(self):
        vocab_file = self.labse_layer.resolved_object.vocab_file.asset_path.numpy()
        do_lower_case = self.labse_layer.resolved_object.do_lower_case.numpy()
        return bert.bert_tokenization.FullTokenizer(vocab_file, do_lower_case)

    def _create_input(self, input_strings):
        input_ids_all, input_mask_all, segment_ids_all = [], [], []
        for input_string in input_strings:
            # Tokenize input.
            input_tokens = ["[CLS]"] + self.tokenizer.tokenize(input_string) + ["[SEP]"]
            input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)
            sequence_length = min(len(input_ids), self.max_seq_length)

            # Padding or truncation.
            if len(input_ids) >= self.max_seq_length:
                input_ids = input_ids[:self.max_seq_length]
            else:
                input_ids = input_ids + [0] * (self.max_seq_length - len(input_ids))

            input_mask = [1] * sequence_length + [0] * (self.max_seq_length - sequence_length)

            input_ids_all.append(input_ids)
            input_mask_all.append(input_mask)
            segment_ids_all.append([0] * self.max_seq_length)
        return np.array(input_ids_all), np.array(input_mask_all), np.array(segment_ids_all)

    def encode(self, input_text):
        if self.labse_model is None:
            self._init_model()
        input_ids, input_mask, segment_ids = self._create_input(input_text)
        return self.labse_model([input_ids, input_mask, segment_ids])

    def batcher(self, params, batch: List[List[str]]) -> np.ndarray:
        batch = [" ".join(sent) if sent != [] else ['.'] for sent in batch]
        embeddings = self.encode(batch)
        return embeddings
