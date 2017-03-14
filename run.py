#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
Q2: Recurrent neural nets for NER
"""

from __future__ import absolute_import
from __future__ import division

import argparse
import logging
import sys
import time
from datetime import datetime
import copy

import tensorflow as tf
import numpy as np

import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from util import print_sentence, write_conll, read_dat, read_lab
from data_util import load_and_preprocess_data, load_embeddings, ModelHelper
from defs import LBLS
from attention_model import AttentionModel
from siamese_model import SiameseModel
#from siamese_distangle_model import SiameseDistangleModel  # TODO
from bow_model import BOWModel
import imp

logger = logging.getLogger("hw3.q2")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


def do_train(args):

    # load config from input
    config_module_name = args.config.split(os.path.sep)[-1]
    config_module = imp.load_source(config_module_name, args.config)
    config = config_module.Config(args)
    print args.config

    add_end_token = args.model is AttentionModel
    helper, train_dat1, train_dat2, train_lab, dev_dat1, dev_dat2, dev_lab = load_and_preprocess_data(args, add_end_token=add_end_token)
    train = zip(train_dat1, train_dat2, train_lab)
    dev = zip(dev_dat1, dev_dat2, dev_lab)
    embeddings = load_embeddings(args, helper)
    config.embed_size = embeddings.shape[1]
    helper.save(config.output_path)

    handler = logging.FileHandler(config.log_output)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logging.getLogger().addHandler(handler)

    report = None #Report(Config.eval_output)

    with tf.Graph().as_default():
        logger.info("Building model...",)
        start = time.time()
        model = args.model(helper, config, embeddings)
        logger.info("took %.2f seconds", time.time() - start)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        with tf.Session() as session:
            session.run(init)
            model.fit(session, saver, train, dev)
            if report:
                report.log_output(model.output(session, dev))
                report.save()
            else:
                # Save predictions in a text file.
                output = model.output(session, dev)
                sentences, labels, predictions = zip(*output)
                output = zip(sentences, labels, predictions)

                with open(model.config.conll_output, 'w') as f:
                    write_conll(f, output)
                with open(model.config.eval_output, 'w') as f:
                    for sentence, labels, predictions in output:
                        print_sentence(f, sentence, labels, predictions)

def do_evaluate(args):

    # load config from input
    config_module_name = args.config.split(os.path.sep)[-1]
    config_module = imp.load_source(config_module_name, args.config)
    config = config_module.Config(args)
    print args.model_path, args.config

    helper = ModelHelper.load(args.model_path)
    dev_q1 = read_dat(args.data_dev1)
    dev_q2 = read_dat(args.data_dev2)
    dev_lab = read_lab(args.data_dev_labels)
    dev_dat1 = helper.vectorize(dev_q1)
    dev_dat2 = helper.vectorize(dev_q2)
    dev_raw = zip(dev_dat1, dev_dat2, dev_lab)

    embeddings = load_embeddings(args, helper)
    config.embed_size = embeddings.shape[1]

    with tf.Graph().as_default():
        logger.info("Building model...",)
        start = time.time()
        model = args.model(helper, config, embeddings)

        logger.info("took %.2f seconds", time.time() - start)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        dev_processed = model.preprocess_sequence_data(dev_raw)
        with tf.Session() as session:
            session.run(init)
            saver.restore(session, model.config.model_output)
            dev_scores = model.evaluate(session, dev_processed, dev_raw)
            print "acc/P/R/F1/loss: %.3f/%.3f/%.3f/%.3f/%.4f" % dev_scores

def do_shell(args):

    # load config from input
    config_module_name = args.config.split(os.path.sep)[-1]
    config_module = imp.load_source(config_module_name, args.config)
    config = config_module.Config(args)
    print args.model_path, args.config

    helper = ModelHelper.load(args.model_path)
    embeddings = load_embeddings(args, helper)
    config.embed_size = embeddings.shape[1]

    with tf.Graph().as_default():
        logger.info("Building model...",)
        start = time.time()
        model = args.model(helper, config, embeddings)
        logger.info("took %.2f seconds", time.time() - start)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        with tf.Session() as session:
            session.run(init)
            saver.restore(session, model.config.model_output)

            print("""Welcome!
You can use this shell to explore the behavior of your model.
Please enter sentences with spaces between tokens, e.g.,
input> Germany 's representative to the European Union 's veterinary committee .
""")
            while True:
                # Create simple REPL
                try:
                    sentence = raw_input("input> ")
                    tokens = sentence.strip().split(" ")
                    for sentence, _, predictions in model.output(session, [(tokens, ["O"] * len(tokens))]):
                        predictions = [LBLS[l] for l in predictions]
                        print_sentence(sys.stdout, sentence, [""] * len(tokens), predictions)
                except EOFError:
                    print("Closing session.")
                    break

def model_class(model_name):
    if model_name == "attention":
        return AttentionModel
    if model_name == "siamese":
        return SiameseModel
    #if model_name == "siamese-distangle":  # TODO
    #    return SiameseDistangleModel
    if model_name == "bow":
        return BOWModel
    raise ValueError("Unknown model: " + model_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trains and tests an NER model')
    subparsers = parser.add_subparsers()

    command_parser = subparsers.add_parser('train', help='')
    command_parser.add_argument('-m', '--model', type=model_class, required=True, help="Model to use.")
    command_parser.add_argument('-dt1', '--data-train1', dest='data_train1', type=argparse.FileType('r'))
    command_parser.add_argument('-dt2', '--data-train2', dest='data_train2', type=argparse.FileType('r'))
    command_parser.add_argument('-dtl', '--data-train-labels', dest='data_train_labels', type=argparse.FileType('r'))
    command_parser.add_argument('-dd1', '--data-dev1', dest='data_dev1', type=argparse.FileType('r'))
    command_parser.add_argument('-dd2', '--data-dev2', dest='data_dev2', type=argparse.FileType('r'))
    command_parser.add_argument('-ddl', '--data-dev-labels', dest='data_dev_labels', type=argparse.FileType('r'))
    command_parser.add_argument('-de1', '--data-test1', dest='data_test1', type=argparse.FileType('r'))
    command_parser.add_argument('-de2', '--data-test2', dest='data_test2', type=argparse.FileType('r'))
    command_parser.add_argument('-v', '--vocab', type=argparse.FileType('r'), default="data/glvocab_1_100.txt", help="Path to vocabulary file")
    command_parser.add_argument('-vv', '--vectors', type=argparse.FileType('r'), default="data/glwordvectors_1_100.txt", help="Path to word vectors file")
    command_parser.add_argument('-eb', '--embed_size', dest='embed_size', default=100)
    command_parser.add_argument('-cfg', '--config', dest='config')
    command_parser.set_defaults(func=do_train)

    command_parser = subparsers.add_parser('evaluate', help='')
    command_parser.add_argument('-m', '--model', type=model_class, required=True, help="Model to use.")
    command_parser.add_argument('-dd1', '--data-dev1', dest='data_dev1', type=argparse.FileType('r'))
    command_parser.add_argument('-dd2', '--data-dev2', dest='data_dev2', type=argparse.FileType('r'))
    command_parser.add_argument('-ddl', '--data-dev-labels', dest='data_dev_labels', type=argparse.FileType('r'))
    command_parser.add_argument('-mp', '--model-path', help="Training data")
    command_parser.add_argument('-v', '--vocab', type=argparse.FileType('r'), default="data/vocab.txt", help="Path to vocabulary file")
    command_parser.add_argument('-vv', '--vectors', type=argparse.FileType('r'), default="data/wordVectors.txt", help="Path to word vectors file")
    command_parser.add_argument('-eb', '--embed_size', dest='embed_size', default=100)
    command_parser.add_argument('-cfg', '--config', dest='config')
    command_parser.set_defaults(func=do_evaluate)

    command_parser = subparsers.add_parser('shell', help='')
    command_parser.add_argument('-m', '--model', type=model_class, required=True, help="Model to use.")
    command_parser.add_argument('-mp', '--model-path', help="Training data")
    command_parser.add_argument('-cfg', '--config', dest='config')
    command_parser.add_argument('-v', '--vocab', type=argparse.FileType('r'), default="data/vocab.txt", help="Path to vocabulary file")
    command_parser.add_argument('-vv', '--vectors', type=argparse.FileType('r'), default="data/wordVectors.txt", help="Path to word vectors file")
    command_parser.set_defaults(func=do_shell)

    ARGS = parser.parse_args()
    if ARGS.func is None:
        parser.print_help()
        sys.exit(1)
    else:
        ARGS.func(ARGS)
