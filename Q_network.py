from ai import AI
from display import CursesDisplay
import tensorflow as tf
import numpy as np
import pickle
import random
#import collections
import curses
import time
from collections import deque

class Q_network(object):

    def __init__(self, board_size, action_size, sess, optimizer, name, variables=None):

        self.sess = sess
        self.optimizer = optimizer
        self.name=name

        self.input_boards = tf.placeholder(tf.float32, shape=[None, board_size])
        self.actions = tf.placeholder(tf.int32, [None])
        self.targets = tf.placeholder(tf.float32, [None])

        n_hidden_1 = 128
        n_hidden_2 = 256
        default_variables = {
            'h1': tf.Variable(tf.random_normal([board_size, n_hidden_1]), name='h1_{}'.format(self.name)),
            'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]), name='h2_{}'.format(self.name)),
            'h_out': tf.Variable(tf.random_normal([n_hidden_2, action_size]), name='hout_{}'.format(self.name)),
            'b1': tf.Variable(tf.random_normal([n_hidden_1]), name='b1_{}'.format(self.name)),
            'b2': tf.Variable(tf.random_normal([n_hidden_2]), name='b2_{}'.format(self.name)),
            'b_out': tf.Variable(tf.random_normal([action_size]), name='bout_{}'.format(self.name))
        }


        if variables==None:
            self.variables = default_variables.copy()
        else:
            self.variables=variables

        # Create model
        # Hidden layer with RELU activation and dropout
        z_1 = tf.add(tf.matmul(self.input_boards, self.variables['h1']), self.variables['b1'])
        layer_1 = tf.layers.dropout(tf.nn.relu(z_1), rate=0.2)
        # Hidden layer with RELU activation
        z_2 = tf.add(tf.matmul(layer_1, self.variables['h2']), self.variables['b2'])
        layer_2 = tf.layers.dropout(tf.nn.relu(z_2), rate=0.2)
        # Output layer with linear activation and dropout
        z_out = tf.add(tf.matmul(layer_2, self.variables['h_out']), self.variables['b_out'])
        self.scores = tf.nn.softmax(z_out)


        actions_scores = tf.reduce_sum(self.scores*tf.one_hot(self.actions, action_size), axis = 1)

        # optimization
        self.loss = tf.reduce_mean(tf.square(actions_scores - self.targets))
        self.train_op = self.optimizer.minimize(self.loss)

    def compute_scores(self, boards):
        return self.sess.run(self.scores, feed_dict={self.input_boards: boards})

    def train(self, boards, actions, targets):
        return self.sess.run([self.loss, self.train_op], feed_dict={self.input_boards:boards, self.actions:actions, self.targets:targets})


#  subroutine to update target network i.e. to copy from principal network to target network
def build_target_update(from_scope, to_scope, b = 0.2):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=to_scope)
    op = []
    for v1, v2 in zip(from_vars, to_vars):
        op.append(v2.assign(v1*b + v2*(1-b)))
    return op



# Implement replay buffer
class ReplayBuffer(object):

    def __init__(self, max_length):
        """
        max_length: max number of tuples to store in the buffer
        if there are more tuples than max_length, pop out the oldest tuples
        """
        self.buffer = deque()
        self.length = 0
        self.max_length = max_length

    def append(self, experience):
        """
        this function implements appending new experience tuple
        """
        self.buffer.append(experience)
        self.length += 1

    def append_frame(self, experience_frame):
        """
        this function implements appending a pandas dataframe of new experience tuples
        """
        actions = experience_frame[126]
        rewards = experience_frame[127]
        boards = experience_frame.drop([126,127], axis=1)
        N = experience_frame.shape[0]
        # each frame correspond to a game
        for i in range(N-1):
            self.buffer.append((boards.loc[i], actions[i], rewards[i], boards.loc[i+1]))
            self.length += 1
        self.buffer.append((boards.loc[N], actions[N], rewards[N], boards.loc[N]))
        self.length += 1

    def pop(self):
        """
        pop out the oldest tuples if self.length > self.max_length
        """
        while self.length > self.max_length:
            self.buffer.popleft()
            self.length -= 1

    def sample(self, batchsize):
        """
        this function samples 'batchsize' experience tuples
        batchsize: size of the minibatch to be sampled
        return: a list of tuples of form (s,a,r,s^\prime)
        """
        return random.sample(self.buffer, batchsize)
