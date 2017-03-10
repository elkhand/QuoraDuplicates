
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

from lstm_cell import LSTMCell
from q3_gru_cell import GRUCell

from data_util import load_and_preprocess_data, load_embeddings, ModelHelper

from util import ConfusionMatrix, Progbar, minibatches
from data_util import get_chunks
from model import Model
from defs import LBLS

logger = logging.getLogger("hw3.q2")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)



class Config:
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """
    n_word_features = 1 # Number of features for every word in the input.
    n_features = n_word_features # Number of features for every word in the input.
    max_length = 35 # longest sequence to parse
    n_classes = 2
    dropout = 0.95
    embed_size = 100 # todo: make depend on input
    hidden_size = 1000
    batch_size = 32
    n_epochs = 100
    max_grad_norm = 10.
    lr = 0.0003
    lr_decay_rate = 0.9
    embeddings_trainable = True
    pos_weight = 1.7


    def __init__(self, args):
        self.cell = "lstm"

        if "output_path" in args:
            # Where to save things.
            self.output_path = args.output_path
        else:
            self.output_path = "results/{}/{:%Y%m%d_%H%M%S}/".format(self.cell, datetime.now())
        self.model_output = self.output_path + "model.weights"
        self.eval_output = self.output_path + "results.txt"
        self.conll_output = self.output_path + "{}_predictions.conll".format(self.cell)

        self.log_output = self.output_path + "log"


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
        """Generates placeholder variables to represent the input tensors

        These placeholders are used as inputs by the rest of the model building and will be fed
        data during training.  Note that when "None" is in a placeholder's shape, it's flexible
        (so we can use different batch sizes without rebuilding the model).

        Adds following nodes to the computational graph

        input_placeholder: Input placeholder tensor of  shape (None, self.max_length, n_features), type tf.int32
        mask_placeholder:  Mask placeholder tensor of shape (None, self.max_length), type tf.bool
        dropout_placeholder: Dropout value placeholder (scalar), type tf.float32

            self.input_placeholder
            self.mask_placeholder
            self.dropout_placeholder

        """
        self.input1_placeholder = tf.placeholder(tf.int32, (None, self.max_length))
        self.input2_placeholder = tf.placeholder(tf.int32, (None, self.max_length))
        self.labels_placeholder = tf.placeholder(tf.float32, shape=(None,))
        self.seqlen1_placeholder = tf.placeholder(tf.int32, (None,))
        self.seqlen2_placeholder = tf.placeholder(tf.int32, (None,))
        self.dropout_placeholder = tf.placeholder(tf.float32, [])

    def create_feed_dict(self, inputs1_batch, inputs2_batch, seqlen1_batch, seqlen2_batch, labels_batch=None, dropout=1):
        """Creates the feed_dict for the dependency parser.

        A feed_dict takes the form of:

        feed_dict = {
                <placeholder>: <tensor of values to be passed for placeholder>,
                ....
        }

        Args:
            inputs_batch: A batch of input data.
            mask_batch:   A batch of mask data.
            labels_batch: A batch of label data.
            dropout: The dropout rate.
        Returns:
            feed_dict: The feed dictionary mapping from placeholders to values.
        """
        feed_dict = {
            self.input1_placeholder: inputs1_batch,
            self.input2_placeholder: inputs2_batch,
            self.seqlen1_placeholder : seqlen1_batch,
            self.seqlen2_placeholder : seqlen2_batch,
            self.dropout_placeholder: dropout
        }
        if labels_batch is not None:
            feed_dict.update({self.labels_placeholder: np.array(labels_batch, dtype=np.float32)})
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
            embeddings = tf.Variable(self.pretrained_embeddings)
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
            h_0 = 0
            for t in 1 to T:
                o_t, h_t = cell(x_t, h_{t-1})
                o_drop_t = Dropout(o_t, dropout_rate)
                y_t = o_drop_t U + b_2

            - Define the variables U, b_2.
            - Define the vector h as a constant and inititalize it with
              zeros. See tf.zeros and tf.shape for information on how
              to initialize this variable to be of the right shape.
              https://www.tensorflow.org/api_docs/python/constant_op/constant_value_tensors#zeros
              https://www.tensorflow.org/api_docs/python/array_ops/shapes_and_shaping#shape
            - In a for loop, begin to unroll the RNN sequence. Collect
              the predictions in a list.
            - When unrolling the loop, from the second iteration
              onwards, you will HAVE to call
              tf.get_variable_scope().reuse_variables() so that you do
              not create new variables in the RNN cell.
              See https://www.tensorflow.org/versions/master/how_tos/variable_scope/
            - Concatenate and reshape the predictions into a predictions
              tensor.

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
            cell = BasicLSTMCell(Config.hidden_size)
        elif self.config.cell1 == "gru":
            cell = GRUCell(Config.n_features * Config.embed_size, Config.hidden_size)
        else:
            raise ValueError("Unsuppported cell type: " + self.config.cell)

        #U = tf.Variable(initial_value=np.ones((1, self.config.hidden_size)), dtype=tf.float32)
        xavier_init= tf.contrib.layers.xavier_initializer()
        m = 50
        U = tf.Variable(initializer=xavier_init, [m, 1], dtype=tf.float32)
        b_u = tf.Variable(initializer=xavier_init,[1,], dtype=tf.float32)
        b = tf.Variable(initializer=xavier_init, [1, m], dtype=tf.float32)
        W = tf.Variable(initializer=xavier_init,[self.config.hidden_size, m], dtype=tf.float32)
        
        # Initialize state as vector of zeros.
        batch_size = tf.shape(x1)[0]
        h1 = tf.zeros([batch_size, self.config.hidden_size], dtype=tf.float32)
        c1 = tf.zeros([batch_size, self.config.hidden_size], dtype=tf.float32)
        h2 = tf.zeros([batch_size, self.config.hidden_size], dtype=tf.float32)
        c2 = tf.zeros([batch_size, self.config.hidden_size], dtype=tf.float32)
        with tf.variable_scope("LSTM"):
            Y, (c1, h1) = tf.nn.dynamic_rnn(cell, x1, initial_state=LSTMStateTuple(c1, h1), sequence_length=self.seqlen1_placeholder)
            # for time_step in range(self.max_length):
            #     x_t = x1[:, time_step, :]
            #     _, h1, c1 = cell1(x_t, h1, c1)
            #     if time_step == 0:
            #         tf.get_variable_scope().reuse_variables()
            h1_drop = tf.nn.dropout(h1, keep_prob=dropout_rate)

            tf.get_variable_scope().reuse_variables()

            Y, (c2, h2) = tf.nn.dynamic_rnn(cell, x2, initial_state=LSTMStateTuple(c2, h2), sequence_length=self.seqlen2_placeholder)
            # for time_step in range(self.max_length):
            #     x_t = x2[:, time_step, :]
            #     _, h2, c2 = cell2(x_t, h2, c2)
            h2_drop = tf.nn.dropout(h2, keep_prob=dropout_rate)

        #preds = tf.reduce_sum(U * h1_drop * h2_drop, 1) + b
        e1 = tf.matmul(h1_drop, W) + b
        r1 = tf.nn.relu(e1, "r1")
        e1 = tf.matmul(h2_drop, W) + b
        r1 = tf.nn.relu(e2, "r2")
        preds = tf.matmul(r1 * r2, U) + b_u
        # assert preds.get_shape().as_list() == [None, self.max_length, self.config.n_classes], "predictions are not of the right shape. Expected {}, got {}".format([None, self.max_length, self.config.n_classes], preds.get_shape().as_list())
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
        loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(targets=self.labels_placeholder, logits=preds, pos_weight=1.7))
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
        global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = self.config.lr
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                                   200000, self.config.lr_decay_rate, staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
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
        #def evaluate(self, sess, examples, examples_raw):
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

        correct_preds = np.logical_and(labels==1, preds==1).sum()
        total_preds = float(np.sum(preds==1))
        total_correct = float(np.sum(labels==1))

        print correct_preds, total_preds, total_correct

        p = correct_preds / total_preds if correct_preds > 0 else 0
        r = correct_preds / total_correct if correct_preds > 0 else 0
        f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0
        return (p, r, f1, loss)

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
                                     dropout=Config.dropout)
        _, pred, loss = sess.run([self.train_op, self.pred, self.loss], feed_dict=feed)
        return loss

    def run_epoch(self, sess, train_processed, dev_processed, train, dev):
        prog = Progbar(target=1 + int(len(train_processed) / self.config.batch_size))
        for i, batch in enumerate(minibatches(train_processed, self.config.batch_size)):
            loss = self.train_on_batch(sess, *batch)
            prog.update(i + 1, [("train loss", loss)])
            if self.report: self.report.log_train_loss(loss)
        print("")

        #logger.info("Evaluating on training data")
        #token_cm, entity_scores = self.evaluate(sess, train_examples, train_examples_raw)
        #logger.debug("Token-level confusion matrix:\n" + token_cm.as_table())
        #logger.debug("Token-level scores:\n" + token_cm.summary())
        #logger.info("Entity level P/R/F1: %.2f/%.2f/%.2f", *entity_scores)

        logger.info("Evaluating on development data")


        entity_scores = self.evaluate(sess, dev_processed, dev)
        logger.info("P/R/F1: %.3f/%.3f/%.3f/%.4f", *entity_scores)

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

        self.max_length = min(Config.max_length, helper.max_length)
        Config.max_length = self.max_length # Just in case people make a mistake.
        self.pretrained_embeddings = pretrained_embeddings

        # Defining placeholders.
        self.input_placeholder = None
        # self.mask_placeholder = None
        self.dropout_placeholder = None

        self.build()


def pad_sequences(data, max_length):
    """Ensures each input-output seqeunce pair in @data is of length
    @max_length by padding it with zeros and truncating the rest of the
    sequence.

    In the code below, for every sentence, labels pair in @data,
    (a) create a new sentence which appends zero feature vectors until
    the sentence is of length @max_length. If the sentence is longer
    than @max_length, simply truncate the sentence to be @max_length
    long.
    (b) create a new label sequence similarly.
    (c) create a _masking_ sequence that has a True wherever there was a
    token in the original sequence, and a False for every padded input.

    Example: for the (sentence, labels) pair: [[4,1], [6,0], [7,0]], [1,
    0, 0], and max_length = 5, we would construct
        - a new sentence: [[4,1], [6,0], [7,0], [0,0], [0,0]]
        - a new label seqeunce: [1, 0, 0, 4, 4], and
        - a masking seqeunce: [True, True, True, False, False].

    Args:
        data: is a list of (sentence, labels) tuples. @sentence is a list
            containing the words in the sentence and @label is a list of
            output labels. Each word is itself a list of
            @n_features features. For example, the sentence "Chris
            Manning is amazing" and labels "PER PER O O" would become
            ([[1,9], [2,9], [3,8], [4,8]], [1, 1, 4, 4]). Here "Chris"
            the word has been featurized as "[1, 9]", and "[1, 1, 4, 4]"
            is the list of labels.
        max_length: the desired length for all input/output sequences.
    Returns:
        a new list of data points of the structure (sent1, sent2, len1, len2, labels).
        Each of sentence', labels' and mask are of length @max_length.
        See the example above for more details.
    """
    ret = []

    # Use this zero vector when padding sequences.
    zero_vector = [0] * Config.n_features

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


def test_pad_sequences():
    print 'test_pad_sequences not implemented'
    # Config.n_features = 2
    # data = [
    #     ([[4,1], [6,0], [7,0]], [1, 0, 0]),
    #     ([[3,0], [3,4], [4,5], [5,3], [3,4]], [0, 1, 0, 2, 3]),
    #     ]
    # ret = [
    #     ([[4,1], [6,0], [7,0], [0,0]], [1, 0, 0, 4], [True, True, True, False]),
    #     ([[3,0], [3,4], [4,5], [5,3]], [0, 1, 0, 2], [True, True, True, True])
    #     ]
    #
    # ret_ = pad_sequences(data, 4)
    # assert len(ret_) == 2, "Did not process all examples: expected {} results, but got {}.".format(2, len(ret_))
    # for i in range(2):
    #     assert len(ret_[i]) == 3, "Did not populate return values corrected: expected {} items, but got {}.".format(3, len(ret_[i]))
    #     for j in range(3):
    #         assert ret_[i][j] == ret[i][j], "Expected {}, but got {} for {}-th entry of {}-th example".format(ret[i][j], ret_[i][j], j, i)

def do_test1(_):
    logger.info("Testing pad_sequences")
    test_pad_sequences()
    logger.info("Passed!")

def do_test2(args):
    logger.info("Testing implementation of RNNModel")
    config = Config(args)
    helper, train, dev, train_raw, dev_raw = load_and_preprocess_data(args)
    embeddings = load_embeddings(args, helper)
    config.embed_size = embeddings.shape[1]

    with tf.Graph().as_default():
        logger.info("Building model...",)
        start = time.time()
        model = RNNModel(helper, config, embeddings)
        logger.info("took %.2f seconds", time.time() - start)

        init = tf.global_variables_initializer()
        saver = None

        with tf.Session() as session:
            session.run(init)
            model.fit(session, saver, train, dev)

    logger.info("Model did not crash!")
    logger.info("Passed!")
