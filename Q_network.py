from ai import AI
from display import CursesDisplay
import tensorflow as tf
import numpy as np
import pickle
import random
import collections
import curses
import time


class Q_network(object):
    def __init__(self, board_size, action_space, sess, optimizer):
        self.sess = sess
        self.optimizer = optimizer
        hid_units_1 = 128
        hid_units_2 = 256
        output_units = 167

        self.input_boards = tf.placeholder(tf.float32, shape=[None, 126])
        self.actions = tf.placeholder(tf.int32, [None])
        self.targets = tf.placeholder(tf.float32, [None])

        # network
        layer_1 = tf.layers.dropout(tf.layers.dense(self.input_boards, hid_units_1, activation=tf.nn.relu), rate=0.2)
        layer_2 = tf.layers.dropout(tf.layers.dense(layer_1, hid_units_2, activation=tf.nn.relu), rate=0.2)
        self.scores = tf.layers.dense(layer_1, action_space, activation=tf.nn.softmax)

        actions_scores = tf.reduce_sum(self.scores*tf.one_hot(self.actions, action_space), axis = 1)

        # optimization
        self.loss = tf.reduce_mean(tf.square(actions_scores - self.targets))
        self.train_op = self.optimizer.minimize(self.loss)


    def compute_scores(self, boards):
        return self.sess.run(self.scores, feed_dict={self.input_boards: boards})

    def train(self, boards, actions, targets):
        return self.sess.run([self.loss, self.train_op], feed_dict={self.input_boards:boards, self.actions:actions, self.targets:targets})
