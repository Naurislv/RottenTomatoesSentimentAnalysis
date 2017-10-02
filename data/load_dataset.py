"""Load dataset and create structure ready for training/testing neural network."""

# Standard imports
import logging
import os
import re

# Dependecy imports
import pandas as pd


def _parse(data):

    data['Phrase'] = data['Phrase'].map(lambda x: x.lower()) # Convert all letters to lower case
    data['Phrase'] = data['Phrase'].map((lambda x: re.sub(r'[^a-zA-z0-9\s]', '', x)))

    return data


def _load():
    # Current script working dir so we could call script from everywhere
    cwd = os.path.dirname(os.path.realpath(__file__))

    # Load datasets
    train = pd.read_csv(cwd + "/train.tsv", header=0, delimiter="\t", quoting=3,
                        dtype={'Sentiment': 'category'})
    test = pd.read_csv(cwd + "/test.tsv", header=0, delimiter="\t", quoting=3)

    logging.info('Train, number of phrases: %d', train["PhraseId"].size)
    logging.info('Test, number of phrases: %d', test["PhraseId"].size)

    train = _parse(train)[["Phrase", "Sentiment"]]
    test = _parse(test)[["PhraseId", "Phrase"]]

    return train, test

TRAIN_SET, TEST_SET = _load()
