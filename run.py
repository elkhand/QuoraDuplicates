#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
Final project: Determining whether two questions are duplicates of each other.
"""

from __future__ import absolute_import
from __future__ import division

import argparse
import logging
import os
import sys
import time

import tensorflow as tf
import numpy as np

from util import write_conll, read_dat, read_lab
from attention_model import AttentionModel
from siamese_model import SiameseModel
from bow_model import BOWModel
import imp
import matplotlib
matplotlib.use('agg')
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
from data_util import *

logger = logging.getLogger("FinalProject")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


def do_train(args):

    # load config from input
    config_module_name = args.config.split(os.path.sep)[-1]
    config_module = imp.load_source(config_module_name, args.config)
    config = config_module.Config(args)
    print args.config

    helper, train_dat1, train_dat2, train_lab, dev_dat1, dev_dat2, dev_lab = load_and_preprocess_data(args, is_train=True)
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
                # output = model.output(session, dev)
                # sentences, labels, predictions = zip(*output)
                # output = zip(sentences, labels, predictions)

                # with open(model.config.conll_output, 'w') as f:
                #     write_conll(f, output)
                # with open(model.config.eval_output, 'w') as f:
                #     for sentence, labels, predictions in output:
                #         print_sentence(f, sentence, labels, predictions)
                pass

def do_evaluate(args):

    # load config from input
    config_module_name = args.config.split(os.path.sep)[-1]
    config_module = imp.load_source(config_module_name, args.config)
    config = config_module.Config(args)
    print args.model_path, args.config

    helper = ModelHelper.load(args.model_path)
    test_q1 = read_dat(args.data_test1)
    test_q2 = read_dat(args.data_test2)

    # add end token
    add_end_token = args.model is AttentionModel
    if add_end_token:
        for i in range(len(test_q1)):
            test_q1[i].append(END_TOKEN)
        for i in range(len(test_q2)):
            test_q2[i].append(END_TOKEN)

    test_lab = read_lab(args.data_test_labels)
    test_dat1 = helper.vectorize(test_q1)
    test_dat2 = helper.vectorize(test_q2)
    test_raw = zip(test_dat1, test_dat2, test_lab)

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
            test_scores = model.evaluate(session, test_raw)
            labels = test_scores[-1]
            preds = test_scores[-2]
            test_scores = test_scores[:5]
            print "acc/P/R/F1/loss: %.3f/%.3f/%.3f/%.3f/%.4f" % test_scores
            outputConfusionMatrix(labels,preds, "confusionMatrix.png")

def do_shell(args):

    # load config from input
    config_module_name = args.config.split(os.path.sep)[-1]
    config_module = imp.load_source(config_module_name, args.config)
    config = config_module.Config(args)
    print args.model_path, args.config

    helper = ModelHelper.load(args.model_path)
    embeddings = load_embeddings(args, helper)
    config.embed_size = embeddings.shape[1]

    add_end_token = args.model is AttentionModel


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
Please enter sentences with spaces between tokens, e.g.,
input1> Do you like cats ?
input2> Are cats better than people ?
""")
            while True:
                # Create simple REPL
                try:
                    sentence1 = raw_input("input1> ")
                    sentence2 = raw_input("input2> ")
                    tokens1 = sentence1.strip().split(" ")
                    tokens2 = sentence2.strip().split(" ")
                    if add_end_token:
                        tokens1.append(END_TOKEN)
                        tokens2.append(END_TOKEN)
                    sentence1, sentence2 = helper.vectorize([tokens1, tokens2])

                    # For the attention model, we will also print the alpha value.
                    if args.model is AttentionModel:
                        extra_fetch = "LSTM_attention/alpha:0"
                    else:
                        extra_fetch = []
                    predictions, _, _, extras = model.output(session, [(sentence1, sentence2, 0)], extra_fetch)

                    prediction = predictions[0]
                    if prediction == 1:
                        print "Duplicate"
                    else:
                        print "Not Duplicate"

                    if args.model is AttentionModel:
                        alphas = extras
                        alpha = alphas[0]
                        alpha = alpha[0:len(tokens2), 0:len(tokens1)]
                        np.set_printoptions(threshold=np.nan, linewidth=1000)
                        print alpha

                        plt.clf()
                        plt.axis('off')

                        rows = tokens2
                        columns = tokens1
                        cell_text = [[''] * len(columns)] * len(rows)
                        cell_colors = plt.cm.Blues(alpha)

                        table = plt.table(cellText=cell_text, cellColours=cell_colors, rowLabels=rows, colLabels=columns, loc='center', colWidths=[0.08] * len(columns))
                        table.auto_set_font_size(False)
                        table.scale(1, 2)
                        for key, cell in table.get_celld().items():
                            cell.set_linewidth(0)

                        plt.savefig("attention.png", bbox_inches = 'tight')

                except EOFError:
                    print("Closing session.")
                    break

def outputConfusionMatrix(labels, preds, filename):
    """ Generate a confusion matrix """
    cm = confusion_matrix(labels, preds, labels=range(2))
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Reds)
    plt.colorbar()
    classes = ["Same", "Different"]
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(filename, bbox_inches = 'tight')


def model_class(model_name):
    if model_name == "attention":
        return AttentionModel
    if model_name == "siamese":
        return SiameseModel
    if model_name == "bow":
        return BOWModel
    raise ValueError("Unknown model: " + model_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trains and tests an NER model')
    subparsers = parser.add_subparsers()

    # Note: we are inputing the test data to the train process for the embeddings only, since we take the union of words in train,dev,test for tok2id.
    # We are not inputing or using the test labels.
    command_parser = subparsers.add_parser('train', help='')
    command_parser.add_argument('-m', '--model', dest='model', type=model_class, required=True, help="Model to use.")
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
    command_parser.add_argument('-eb', '--embed_size', dest='embed_size', type=int, default=100)
    command_parser.add_argument('-cfg', '--config', required=True)
    command_parser.set_defaults(func=do_train)

    command_parser = subparsers.add_parser('evaluate', help='')
    command_parser.add_argument('-m', '--model', dest='model', type=model_class, required=True, help="Model to use.")
    command_parser.add_argument('-de1', '--data-test1', dest='data_test1', type=argparse.FileType('r'))
    command_parser.add_argument('-de2', '--data-test2', dest='data_test2', type=argparse.FileType('r'))
    command_parser.add_argument('-ddl', '--data-test-labels', dest='data_test_labels', type=argparse.FileType('r'))
    command_parser.add_argument('-mp', '--model-path', required=True, help="Training data")
    command_parser.add_argument('-v', '--vocab', type=argparse.FileType('r'), default="data/vocab.txt", help="Path to vocabulary file")
    command_parser.add_argument('-vv', '--vectors', type=argparse.FileType('r'), default="data/wordVectors.txt", help="Path to word vectors file")
    command_parser.add_argument('-eb', '--embed_size', dest='embed_size', type=int, default=100)
    command_parser.add_argument('-cfg', '--config', required=True)
    command_parser.set_defaults(func=do_evaluate)

    command_parser = subparsers.add_parser('shell', help='')
    command_parser.add_argument('-m', '--model', dest='model', type=model_class, required=True, help="Model to use.")
    command_parser.add_argument('-mp', '--model-path', required=True, help="Training data")
    command_parser.add_argument('-v', '--vocab', type=argparse.FileType('r'), default="data/vocab.txt", help="Path to vocabulary file")
    command_parser.add_argument('-vv', '--vectors', type=argparse.FileType('r'), default="data/wordVectors.txt", help="Path to word vectors file")
    command_parser.add_argument('-eb', '--embed_size', dest='embed_size', type=int, default=100)
    command_parser.add_argument('-cfg', '--config', required=True)
    command_parser.set_defaults(func=do_shell)

    ARGS = parser.parse_args()
    if ARGS.func is None:
        parser.print_help()
        sys.exit(1)
    else:
        ARGS.func(ARGS)
