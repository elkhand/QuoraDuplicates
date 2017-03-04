#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q2: Recurrent neural nets for NER
"""

from __future__ import absolute_import
from __future__ import division

import argparse
import logging
import sys

import tensorflow as tf
import numpy as np

logger = logging.getLogger("hw3.q2.1")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

__author__ = 'kyu'

class LSTMCell(tf.nn.rnn_cell.RNNCell):
    """Wrapper around our RNN cell implementation that allows us to play
    nicely with TensorFlow.
    """
    def __init__(self, input_size, state_size):
        self.input_size = input_size
        self._state_size = state_size

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._state_size

    def __call__(self, inputs, state, cand, scope=None):
        """Updates the state using the previous @state and @inputs.
        The LSTM equations are:

        z_t = sigmoid(x_t U_z + h_{t-1} W_z + b_z)
        r_t = sigmoid(x_t U_r + h_{t-1} W_r + b_r)
        o_t = tanh(x_t U_o + r_t * h_{t-1} W_o + b_o)
        h_t = z_t * h_{t-1} + (1 - z_t) * o_t

        Args:
            inputs: is the input vector of size [None, self.input_size]
            state: is the previous state vector of size [None, self.state_size]
            scope: is the name of the scope to be used when defining the variables inside.
        Returns:
            a pair of the output vector and the new state vector.
        """
        scope = scope or type(self).__name__

        # It's always a good idea to scope variables in functions lest they
        # be defined elsewhere!
        with tf.variable_scope(scope):
            ### YOUR CODE HERE (~6-10 lines)
            xavier_init = tf.contrib.layers.xavier_initializer()

            W_i = tf.get_variable("W_i", [self._state_size, self._state_size], initializer=xavier_init)
            b_i = tf.get_variable("b_i", initializer =np.array(np.zeros([1, self._state_size]), dtype=np.float32))
            W_f = tf.get_variable("W_f", [self._state_size, self._state_size], initializer=xavier_init)
            b_f = tf.get_variable("b_f", initializer =np.array(np.zeros([1, self._state_size]), dtype=np.float32))
            W_o = tf.get_variable("W_o", [self._state_size, self._state_size], initializer=xavier_init)
            b_o = tf.get_variable("b_o", initializer =np.array(np.zeros([1, self._state_size]), dtype=np.float32))
            W_c = tf.get_variable("W_c", [self._state_size, self._state_size], initializer=xavier_init)
            b_c = tf.get_variable("b_c", initializer =np.array(np.zeros([1, self._state_size]), dtype=np.float32))

            i_t = tf.sigmoid(tf.matmul(inputs, W_i) + tf.matmul(state, W_i) + b_i)
            f_t = tf.sigmoid(tf.matmul(inputs, W_f) + tf.matmul(state, W_f) + b_f)
            o_t = tf.sigmoid(tf.matmul(inputs, W_o) + tf.matmul(state, W_o) + b_o)
            c_t = f_t * cand + i_t * tf.tanh(tf.matmul(inputs, W_c) + tf.matmul(state, W_c) + b_c)
            h_t = o_t * tf.tanh(c_t)

            ### END YOUR CODE ###
        # For an RNN , the output and state are the same (N.B. this
        # isn't true for an LSTM, though we aren't using one of those in
        # our assignment)
        new_state = h_t
        cand = c_t
        output = new_state
        return output, new_state, cand

# def test_rnn_cell():
#     with tf.Graph().as_default():
#         with tf.variable_scope("test_rnn_cell"):
#             x_placeholder = tf.placeholder(tf.float32, shape=(None,3))
#             h_placeholder = tf.placeholder(tf.float32, shape=(None,2))
#
#             with tf.variable_scope("rnn"):
#                 tf.get_variable("W_x", initializer=np.array(np.eye(3,2), dtype=np.float32))
#                 tf.get_variable("W_h", initializer=np.array(np.eye(2,2), dtype=np.float32))
#                 tf.get_variable("b",  initializer=np.array(np.ones(2), dtype=np.float32))
#
#             tf.get_variable_scope().reuse_variables()
#             cell = LSTMCell(3, 2)
#             y_var, ht_var = cell(x_placeholder, h_placeholder, scope="rnn")
#
#             init = tf.global_variables_initializer()
#             with tf.Session() as session:
#                 session.run(init)
#                 x = np.array([
#                     [0.4, 0.5, 0.6],
#                     [0.3, -0.2, -0.1]], dtype=np.float32)
#                 h = np.array([
#                     [0.2, 0.5],
#                     [-0.3, -0.3]], dtype=np.float32)
#                 y = np.array([
#                     [0.832, 0.881],
#                     [0.731, 0.622]], dtype=np.float32)
#                 ht = y
#
#                 y_, ht_ = session.run([y_var, ht_var], feed_dict={x_placeholder: x, h_placeholder: h})
#                 print("y_ = " + str(y_))
#                 print("ht_ = " + str(ht_))
#
#                 assert np.allclose(y_, ht_), "output and state should be equal."
#                 assert np.allclose(ht, ht_, atol=1e-2), "new state vector does not seem to be correct."
#
# def do_test(_):
#     logger.info("Testing rnn_cell")
#     test_rnn_cell()
#     logger.info("Passed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tests the LSTM cell')
    # subparsers = parser.add_subparsers()
    #
    # command_parser = subparsers.add_parser('test', help='')
    # command_parser.set_defaults(func=do_test)
    #
    # ARGS = parser.parse_args()
    # if ARGS.func is None:
    #     parser.print_help()
    #     sys.exit(1)
    # else:
    #     ARGS.func(ARGS)
