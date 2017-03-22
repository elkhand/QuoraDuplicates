#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions to process data.
"""
import os
import pickle
import logging
from collections import Counter

import numpy as np
from util import read_dat, read_lab, ConfusionMatrix

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

NONE = "O"
NUM = "NNNUMMM"
UNK = "UUUNKKK"
START_TOKEN = "<s>"
END_TOKEN = "</s>"

def normalize(word):
    """
    Normalize words that are numbers or have casing.
    """
    if word.isdigit(): return NUM
    else: return word.lower()

class ModelHelper(object):
    """
    This helper takes care of preprocessing data, constructing embeddings, etc.
    """
    def __init__(self, tok2id, max_length):
        self.tok2id = tok2id
        self.START = [tok2id[START_TOKEN]]
        self.END = [tok2id[END_TOKEN]]
        self.max_length = max_length

    def vectorize_example(self, sentence, is_train, labels=None):
        unknown_id = self.tok2id[UNK]
        if is_train:
            unk_prob = 0.01
            np.random.seed(0)
            sentence_ = list()
            unk_samp = np.random.random(len(sentence))
            for i,word in enumerate(sentence):
                if unk_samp[i] < unk_prob:
                    sentence_.append(unknown_id)
                else:
                    sentence_.append(self.tok2id.get(normalize(word), unknown_id))
        else:
            sentence_ = [self.tok2id.get(normalize(word), unknown_id) for word in sentence]
        return sentence_

    def vectorize(self, data, is_train=False):
        return [self.vectorize_example(sentence, is_train=is_train) for sentence in data]

    @classmethod
    def build(cls, data):
        # Preprocess data to construct an embedding
        # Reserve 0 for the special NIL token.
        tok2id = build_dict((normalize(word) for sent1, sent2, _ in data for word in sent1 + sent2), offset=1, max_words=1000000)
        tok2id.update(build_dict([START_TOKEN, END_TOKEN, UNK], offset=len(tok2id)))
        assert sorted(tok2id.items(), key=lambda t: t[1])[0][1] == 1
        logger.info("Built dictionary for %d features.", len(tok2id))

        max_length = max(max(len(sent1), len(sent2)) for sent1, sent2, _ in data)

        return cls(tok2id, max_length)

    def save(self, path):
        # Make sure the directory exists.
        if not os.path.exists(path):
            os.makedirs(path)
        # Save the tok2id map.
        with open(os.path.join(path, "features.pkl"), "w") as f:
            pickle.dump([self.tok2id, self.max_length], f)

    @classmethod
    def load(cls, path):
        # Make sure the directory exists.
        assert os.path.exists(path) and os.path.exists(os.path.join(path, "features.pkl"))
        # Save the tok2id map.
        with open(os.path.join(path, "features.pkl")) as f:
            tok2id, max_length = pickle.load(f)
        return cls(tok2id, max_length)

def load_and_preprocess_data(args, add_end_token=False, is_train=False):
    logger.info("Loading training data...")
    train_q1 = read_dat(args.data_train1)
    train_q2 = read_dat(args.data_train2)
    train_lab = read_lab(args.data_train_labels)
    assert len(train_q1) == len(train_q2)
    assert len(train_q1) == len(train_lab)
    logger.info("Done. Read %d sentence pairs", len(train_lab))
    logger.info("Loading dev data...")
    dev_q1 = read_dat(args.data_dev1)
    dev_q2 = read_dat(args.data_dev2)
    dev_lab = read_lab(args.data_dev_labels)
    assert len(dev_q1) == len(dev_q2)
    assert len(dev_q1) == len(dev_lab)
    logger.info("Done. Read %d sentence pairs", len(dev_lab))
    test_q1 = read_dat(args.data_test1)
    test_q2 = read_dat(args.data_test2)
    assert len(test_q1) == len(test_q2)
    test_lab_dummy = [0 for i in range(len(test_q1))]

    train_to_build_lkp = zip(train_q1+dev_q1+test_q1, train_q2+dev_q2+test_q2, train_lab+dev_lab+test_lab_dummy)

    helper = ModelHelper.build(train_to_build_lkp)

    if add_end_token:
        for i in range(len(train_q1)):
            train_q1[i].append(END_TOKEN)
            train_q2[i].append(END_TOKEN)
        for i in range(len(dev_q1)):
            dev_q1[i].append(END_TOKEN)
            dev_q2[i].append(END_TOKEN)


    # now process all the input data.
    train_dat1 = helper.vectorize(train_q1, is_train=is_train)
    train_dat2 = helper.vectorize(train_q2, is_train=is_train)
    dev_dat1   = helper.vectorize(dev_q1)
    dev_dat2   = helper.vectorize(dev_q2)

    return helper, train_dat1, train_dat2, train_lab, dev_dat1, dev_dat2, dev_lab

def load_embeddings(args, helper):
    np.random.seed(0)
    embeddings = np.array(np.random.randn(len(helper.tok2id) + 1, args.embed_size), dtype=np.float32)
    embeddings[0] = 0.
    try:
        for word, vec in zip(args.vocab, args.vectors):
            word = normalize(word.strip())
            if word in helper.tok2id:
                vec = np.array(map(float, vec.split()))
                embeddings[helper.tok2id[word]] = vec
    except:
        args.vectors.seek(0) # Go to the beginning of the file, because "zip" has exhausted the stream.
        for word_and_vec in args.vectors:
            word, vec = word_and_vec.split(None, 1)
            word = normalize(word.strip())
            if word in helper.tok2id:
                vec = np.array(map(float, vec.split()))
                embeddings[helper.tok2id[word]] = vec
    logger.info("Initialized embeddings.")

    return embeddings

def build_dict(words, max_words=None, offset=0):
    cnt = Counter(words)
    if max_words:
        words = cnt.most_common(max_words)
    else:
        words = cnt.most_common()
    return {word: offset+i for i, (word, _) in enumerate(words)}
