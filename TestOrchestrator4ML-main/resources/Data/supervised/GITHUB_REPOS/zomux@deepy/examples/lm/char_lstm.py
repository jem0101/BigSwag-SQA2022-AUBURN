#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import logging
from argparse import ArgumentParser

from vocab import Vocab
from lmdataset import LMDataset
from lm import NeuralLM
from deepy.dataset import SequentialMiniBatches
from deepy.trainers import SGDTrainer, LearningRateAnnealer
from deepy.layers import LSTM, Dense


logging.basicConfig(level=logging.INFO)

resource_dir = os.path.abspath(os.path.dirname(__file__)) + os.sep + "resources"

vocab_path = os.path.join(resource_dir, "ptb.train.txt")
train_path = os.path.join(resource_dir, "ptb.train.txt")
valid_path = os.path.join(resource_dir, "ptb.valid.txt")
vocab = Vocab(char_based=True)
vocab.load(vocab_path, max_size=1000)

model = NeuralLM(input_dim=vocab.size, input_tensor=3)
model.stack(LSTM(hidden_size=100, output_type="sequence"),
            Dense(vocab.size, activation="softmax"))

default_model = os.path.join(os.path.dirname(__file__), "models", "char_lstm_model1.gz")

if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument("--model", default=default_model)
    ap.add_argument("--sample", default="")
    args = ap.parse_args()

    if os.path.exists(args.model):
        model.load_params(args.model)

    lmdata = LMDataset(vocab, train_path, valid_path, history_len=30, char_based=True, max_tokens=300)
    batch = SequentialMiniBatches(lmdata, batch_size=20)

    trainer = SGDTrainer(model)
    annealer = LearningRateAnnealer()

    trainer.run(batch, epoch_controllers=[annealer])

    model.save_params(args.model)
