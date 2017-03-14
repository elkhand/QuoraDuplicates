
from __future__ import absolute_import
from __future__ import division
import logging
import sys
import time
from datetime import datetime
import copy

import tensorflow as tf
import numpy as np

import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_util import load_and_preprocess_data, load_embeddings, ModelHelper

from util import ConfusionMatrix, Progbar, minibatches
from model import Model
from defs import LBLS

logger = logging.getLogger("hw3.q2")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


class RNNModel(Model):
    """
    Implements a recursive neural network with an embedding layer and
    single hidden layer.
    """
    def build(self):
        super(RNNModel, self).build()
        pos_thres = tf.constant(0.5, dtype=tf.float32, shape=(1,))
        self.predictions = tf.greater(tf.sigmoid(self.pred), pos_thres)

    def add_placeholders(self):
        self.input1_placeholder = tf.placeholder(tf.int32, (None, self.max_length))
        self.input2_placeholder = tf.placeholder(tf.int32, (None, self.max_length))
        self.labels_placeholder = tf.placeholder(tf.float32, shape=(None,2))
        self.seqlen1_placeholder = tf.placeholder(tf.int32, (None,))
        self.seqlen2_placeholder = tf.placeholder(tf.int32, (None,))
        self.dropout_placeholder = tf.placeholder(tf.float32, [])

    def create_feed_dict(self, inputs1_batch, inputs2_batch, seqlen1_batch, seqlen2_batch, labels_batch=None, dropout=1):
        feed_dict = {
            self.input1_placeholder: inputs1_batch,
            self.input2_placeholder: inputs2_batch,
            self.seqlen1_placeholder : seqlen1_batch,
            self.seqlen2_placeholder : seqlen2_batch,
            self.dropout_placeholder: dropout
        }
        if labels_batch is not None:
            sp_labels = []
            for l in labels_batch:
                if l==0:
                    sp_labels.append([1,0])
                else:
                    sp_labels.append([0,1])
            sp_labels= np.array(sp_labels,dtype=np.float32)
            feed_dict.update({self.labels_placeholder: sp_labels})
        return feed_dict

    def add_embedding(self, ind):
        """Adds an embedding layer that maps from input tokens (integers) to vectors and then
        concatenates those vectors:

            - Create an embedding tensor and initialize it with self.pretrained_embeddings.
            - Use the input_placeholder to index into the embeddings tensor, resulting in a
              tensor of shape (None, max_length, n_features, embed_size).
            - Concatenates the embeddings by reshaping the embeddings tensor to shape
              (None, max_length, n_features * embed_size).

        Returns:
            embeddings: tf.Tensor of shape (None, max_length, n_features*embed_size)
        """
        if self.config.embeddings_trainable:
            embeddings = tf.Variable(self.pretrained_embeddings, name="embeddings")
        else:
            embeddings = self.pretrained_embeddings

        if ind==1:
            to_concat = tf.nn.embedding_lookup(embeddings, self.input1_placeholder)
        if ind==2:
            to_concat = tf.nn.embedding_lookup(embeddings, self.input2_placeholder)
        embeddings = tf.reshape(to_concat, [-1, self.config.max_length, self.config.n_features* self.config.embed_size])
        return embeddings

    def add_prediction_op(self):
        """Adds the unrolled RNN:

        Remember:
            * Use the xavier initilization for matrices.
            * Note that tf.nn.dropout takes the keep probability (1 - p_drop) as an argument.
            The keep probability should be set to the value of self.dropout_placeholder

        Returns:
            pred: tf.Tensor of shape (batch_size, max_length, n_classes)
        """

        x1 = self.add_embedding(1)
        x2 = self.add_embedding(2)
        dropout_rate = self.dropout_placeholder

        BasicLSTMCell = tf.contrib.rnn.BasicLSTMCell if hasattr(tf.contrib.rnn, 'BasicLSTMCell') else tf.nn.rnn_cell.BasicLSTMCell
        LSTMStateTuple = tf.contrib.rnn.LSTMStateTuple if hasattr(tf.contrib.rnn, 'LSTMStateTuple') else tf.nn.rnn_cell.LSTMStateTuple


        if self.config.cell == "lstm":
            cell = BasicLSTMCell(self.config.hidden_size)
            if hasattr(tf.contrib.rnn, 'DropoutWrapper'):
                cell = tf.contrib.rnn.DropoutWrapper(cell, dropout_rate)
            else:
                cell = tf.nn.rnn_cell.DropoutWrapper(cell, dropout_rate)
        else:
            raise ValueError("Unsuppported cell type: " + self.config.cell)

        xavier_init = tf.contrib.layers.xavier_initializer()

        
        m = self.config.second_hidden_size
        with tf.variable_scope("HiddenLayerVars"):
            b1 = tf.get_variable("b1",initializer=xavier_init,  shape=[1, m])
            b2 = tf.get_variable("b2",initializer=xavier_init,  shape=[1,2])
            W1 = tf.get_variable("W1",initializer=xavier_init, shape=[3*self.config.hidden_size+1, m])
            W2 = tf.get_variable("W2",initializer=xavier_init, shape=[m, 2])
            tf.get_variable_scope().reuse_variables()

        # Initialize state as vector of zeros.
        batch_size = tf.shape(x1)[0]
        h1 = tf.zeros([batch_size, self.config.hidden_size], dtype=tf.float32)
        c1 = tf.zeros([batch_size, self.config.hidden_size], dtype=tf.float32)
        h2 = tf.zeros([batch_size, self.config.hidden_size], dtype=tf.float32)
        c2 = tf.zeros([batch_size, self.config.hidden_size], dtype=tf.float32)
        with tf.variable_scope("LSTM"):
            Y, (c1, h1) = tf.nn.dynamic_rnn(cell, x1, initial_state=LSTMStateTuple(c1, h1), sequence_length=self.seqlen1_placeholder)
            #h1_drop = tf.nn.dropout(h1, keep_prob=dropout_rate)
            tf.get_variable_scope().reuse_variables()
            Y, (c2, h2) = tf.nn.dynamic_rnn(cell, x2, initial_state=LSTMStateTuple(c2, h2), sequence_length=self.seqlen2_placeholder)
            #h2_drop = tf.nn.dropout(h2, keep_prob=dropout_rate)

        h_sub = tf.subtract(h1, h2)
        sqdiff_12 = tf.square(h_sub)
        sqdist_12 = tf.reduce_sum(sqdiff_12, 1)
        h_dist = tf.reshape(sqdist_12, [batch_size,1])
        # mul(h1,h2)
        h_mul =  tf.multiply(h1 , h2) 
        #h_combined = tf.concat(1,[h1,h2,h_dist,h_mul])#3*hidden_size+1
        if int(tf.__version__.split('.')[0]) >= 1: # TensorFlow 1.0 or greater
            h_combined = tf.concat([h1, h2, h_dist, h_mul], 1) # 3*hidden_size+1
        else:
            h_combined = tf.concat(1, [h1, h2, h_dist, h_mul]) # 3*hidden_size+1
        h_combined_drop = tf.nn.dropout(h_combined, keep_prob=dropout_rate)

        e1 = tf.matmul(h_combined, W1) + b1 # [bath_size,m]
        e1_relu = tf.nn.relu(e1)
        e1_drop = tf.nn.dropout(e1_relu, keep_prob=dropout_rate)
        preds = tf.matmul(e1_drop,W2) + b2

        return preds

    def add_loss_op(self, preds):
        """Adds Ops for the loss function to the computational graph.

        Compute averaged cross entropy loss for the predictions.
        Importantly, you must ignore the loss for any masked tokens.

        Args:
            pred: A tensor of shape (batch_size, max_length, n_classes) containing the output of the neural
                  network before the softmax layer.
        Returns:
            loss: A 0-d tensor (scalar)
        """
        
        m = self.config.second_hidden_size
        xavier_init = tf.contrib.layers.xavier_initializer()
        loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(targets=self.labels_placeholder, logits=preds, pos_weight=1.675))       
        with tf.variable_scope("HiddenLayerVars", reuse=True):
            b1 = tf.get_variable("b1",initializer=xavier_init,  shape=[1, m])
            b2 = tf.get_variable("b2",initializer=xavier_init,  shape=[1,2])
            W1 = tf.get_variable("W1",initializer=xavier_init, shape=[3*self.config.hidden_size+1, m])
            W2 = tf.get_variable("W2",initializer=xavier_init, shape=[m, 2])
            tf.get_variable_scope().reuse_variables()
        loss = loss + self.config.beta*tf.nn.l2_loss(W1)+ self.config.beta*tf.nn.l2_loss(W2)

        return loss

    def add_training_op(self, loss):
        """Sets up the training Ops.

        Creates an optimizer and applies the gradients to all trainable variables.
        The Op returned by this function is what must be passed to the
        `sess.run()` call to cause the model to train. See

        https://www.tensorflow.org/versions/r0.7/api_docs/python/train.html#Optimizer

        for more information.

        Use tf.train.AdamOptimizer for this model.
        Calling optimizer.minimize() will return a train_op object.

        Args:
            loss: Loss tensor, from cross_entropy_loss.
        Returns:
            train_op: The Op for training.
        """
        global_step = tf.Variable(0, name='global_step', trainable=False)
        starter_learning_rate = self.config.lr
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                                   85015, self.config.lr_decay_rate, staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_op = optimizer.minimize(loss, global_step=global_step)
        return train_op

    def preprocess_sequence_data(self, examples):
        return pad_sequences(examples, self.max_length)


    def predict_on_batch(self, sess, inputs1_batch, inputs2_batch, seqlen1_batch, seqlen2_batch, labels_batch):
        inputs1_batch = np.array(inputs1_batch)
        inputs2_batch = np.array(inputs2_batch)
        feed = self.create_feed_dict(inputs1_batch=inputs1_batch, inputs2_batch=inputs2_batch, seqlen1_batch=seqlen1_batch, seqlen2_batch=seqlen2_batch, labels_batch=labels_batch)

        logits, predictions, loss = sess.run([self.pred, self.predictions, self.loss], feed_dict=feed)
        return logits, predictions, loss

    def evaluate(self, sess, examples, examples_raw):
        """Evaluates model performance on @examples.

        This function uses the model to predict labels for @examples and constructs a confusion matrix.

        Args:
            sess: the current TensorFlow session.
            examples: A list of vectorized input/output pairs. Examples is padded.
            examples: A list of the original input/output sequence pairs. Raw input,un-processed.
        Returns:
            The F1 score for predicting tokens as named entities.
        """

        (labels, preds, logits), loss = self.output(sess, examples_raw, examples) #*
        labels, preds, logits = np.array(labels, dtype=np.float32), np.array(preds), np.array(logits)
        all_pos = [x for x in preds[:,1]]
        preds = np.array(all_pos)
        correct_preds = np.logical_and(labels==1, preds==1).sum()
        total_preds = float(np.sum(preds==1))
        total_correct = float(np.sum(labels==1))

        print "Correct_preds: ",correct_preds,"\tTotal_preds: ", total_preds,"\tTotal_correct: ", total_correct

        p = correct_preds / total_preds if correct_preds > 0 else 0
        r = correct_preds / total_correct if correct_preds > 0 else 0
        f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0
        acc = sum(labels==preds) / float(len(labels))
        return (acc, p, r, f1, loss)

    def consolidate_predictions(self, examples_raw, examples_processed, preds, logits):
        """Batch the predictions into groups of sentence length.
        """
        assert len(examples_raw) == len(examples_processed)
        assert len(examples_raw) == len(preds)

        labels = [x[2] for x in examples_raw]

        return labels, preds, logits


    def output(self, sess, inputs_raw, inputs):
        """
        Reports the output of the model on examples (uses helper to featurize each example).
        """
        if inputs is None:
            inputs = self.preprocess_sequence_data(self.helper.vectorize(inputs_raw))

        preds = []
        logits = []
        loss_record = []
        prog = Progbar(target=1 + int(len(inputs) / self.config.batch_size))
        for i, batch in enumerate(minibatches(inputs, self.config.batch_size, shuffle=False)):
            # batch = batch[:4] # ignore label
            logits_, preds_, loss_ = self.predict_on_batch(sess, *batch)
            preds += list(preds_)
            logits += list(logits_)
            loss_record.append(loss_)
            prog.update(i + 1, [])
        return self.consolidate_predictions(inputs_raw, inputs, preds, logits), np.mean(loss_record)

    def train_on_batch(self, sess, inputs1_batch, inputs2_batch, seqlen1_batch, seqlen2_batch, labels_batch):
        feed = self.create_feed_dict(inputs1_batch, inputs2_batch, seqlen1_batch, seqlen2_batch, labels_batch=labels_batch,
                                     dropout=self.config.dropout)
        _, pred, loss = sess.run([self.train_op, self.pred, self.loss], feed_dict=feed)
        return loss

    def run_epoch(self, sess, train_processed, dev_processed, train, dev):

        prog = Progbar(target=1 + int(len(train_processed) / self.config.batch_size))
        for i, batch in enumerate(minibatches(train_processed, self.config.batch_size)):
            loss = self.train_on_batch(sess, *batch)
            prog.update(i + 1, [("train loss", loss)])

            if self.report: self.report.log_train_loss(loss)
        print("")

        logger.info("Evaluating on training data: 10k sample")
        n_train_evaluate = 10000
        train_entity_scores = self.evaluate(sess, train_processed[:n_train_evaluate], train[:n_train_evaluate])
        logger.info("acc/P/R/F1/loss: %.3f/%.3f/%.3f/%.3f/%.4f", *train_entity_scores)

        logger.info("Evaluating on development data")
        entity_scores = self.evaluate(sess, dev_processed, dev)
        logger.info("acc/P/R/F1/loss: %.3f/%.3f/%.3f/%.3f/%.4f", *entity_scores)

        # with open(self.config.eval_output, 'a') as f:
        #     f.write('%.4f %.4f %.3f %.3f %.3f %.3f %.3f\n' % (train_entity_scores[4], entity_scores[4], train_entity_scores[3], entity_scores[0], entity_scores[1], entity_scores[2], entity_scores[3]))

        with open(self.config.eval_output, 'a') as f:
            f.write('%.4f %.4f %.3f %.3f %.3f %.3f %.3f %.3f %.3f\n' % (train_entity_scores[4], entity_scores[4], train_entity_scores[0], entity_scores[0], train_entity_scores[3], entity_scores[3], entity_scores[0], entity_scores[1], entity_scores[2]))

        f1 = entity_scores[-2]
        return f1

    def fit(self, sess, saver, train, dev):
        best_score = 0.

        # Padded sentences
        train_processed = self.preprocess_sequence_data(train) # sent1, sent2, label
        dev_processed = self.preprocess_sequence_data(dev)

        for epoch in range(self.config.n_epochs):
            logger.info("Epoch %d out of %d", epoch + 1, self.config.n_epochs)
            score = self.run_epoch(sess, train_processed, dev_processed, train, dev)
            if score > best_score:
                best_score = score
                if saver:
                    logger.info("New best score! Saving model in %s", self.config.model_output)
                    saver.save(sess, self.config.model_output)
            print("")
            if self.report:
                self.report.log_epoch()
                self.report.save()
        return best_score


    def __init__(self, helper, config, pretrained_embeddings, report=None):

        self.helper = helper
        self.config = config
        self.report = report

        self.max_length = min(self.config.max_length, helper.max_length)
        self.config.max_length = self.max_length # Just in case people make a mistake.
        self.pretrained_embeddings = pretrained_embeddings

        # Defining placeholders.
        self.input_placeholder = None
        # self.mask_placeholder = None
        self.dropout_placeholder = None

        self.build()


def pad_sequences(data, max_length, n_features=1):
    ret = []

    # Use this zero vector when padding sequences.
    zero_vector = [0] * n_features

    for sentence1, sentence2, label in data:
        feat_sent1 = zero_vector * max_length
        feat_sent2 = zero_vector * max_length
        for i, word in enumerate(sentence1):
            if i >= max_length:
                break
            feat_sent1[i] = word

        for i, word in enumerate(sentence2):
            if i >= max_length:
                break
            feat_sent2[i] = word
        seqlen1 = min(len(sentence1), max_length)
        seqlen2 = min(len(sentence2), max_length)
        ret.append((feat_sent1, feat_sent2, seqlen1, seqlen2, label))
    return ret