from ai import AI
from display import CursesDisplay
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
import random
import collections
import curses
import time
from Q_network import *
import os
import sys
import subprocess
import argparse

import pyrisk_callable


def main(params):
    NAME = params['ai_name']
    global_episode=params['restart_episode']

    # initialize
    board_size = 3*42 #3*len(territories) # we assume 3 players
    action_size = 167 #len(possible_attacks)
    learning_rate =0.001
    gamma = .95  # discount
    max_length=200000 # for the buffer
    n_training_steps = 10
    batchsize = 32


    buffer = ReplayBuffer(max_length)
    sess = tf.Session()
    # optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate)


    model_path = 'model_files/{}'.format(NAME)
    os.makedirs(model_path, exist_ok=True)


    if global_episode==0:
        # initialize network
        with tf.variable_scope("main"):
            Q_main = Q_network(board_size, action_size, sess, optimizer, 'main')
        with tf.variable_scope("target"):
            Q_target = Q_network(board_size, action_size, sess, optimizer, 'target')

        sess.run(tf.global_variables_initializer())
        #save model
        model_saver = tf.train.Saver(max_to_keep=1)
        model_saver.save(sess, '{}/episode_{}/model'.format(model_path,global_episode))

    else :
        loader = tf.train.import_meta_graph('{}/episode_{}/model.meta'.format(model_path,global_episode))
        loader.restore(sess, '{}/episode_{}/model'.format(model_path,global_episode))
        graph = tf.get_default_graph()
        graph.get_collection('trainable_variables')

        main_vars = {}
        for var in graph.get_collection('trainable_variables'):
            if var.name[:4]=='main':
                ref = var.name.split('/')[1][:-7]
                if ref in ['h1','h2','hout','b1','b2','bout']: # security check
                    main_vars.update({ref: tf.identity(var, name="{}_main".format(ref)) })

        target_vars = {}
        for var in graph.get_collection('trainable_variables'):
            if var.name[:6]=='target':
                ref = var.name.split('/')[1][:-9]
                if ref in ['h1','h2','hout','b1','b2','bout']: # security check
                    target_vars.update({ref: tf.identity(var, name="{}_target".format(ref)) })


        # initialize network
        with tf.variable_scope("main"):
            Q_main = Q_network(board_size, action_size, sess, optimizer, 'main', variables= main_vars)
        with tf.variable_scope("target"):
            Q_target = Q_network(board_size, action_size, sess, optimizer, 'target', variables= target_vars)

        model_saver = tf.train.Saver(max_to_keep=1)


    # build ops
    update = build_target_update("main", "target")  # call sess.run(update) to copy from principal to target

    sess.run(update)


    n_training_steps=10
    n_episodes = 5000

    for _ in range(n_episodes):
        global_episode+=1
        # save model every 100 episodes = every 1000 games
        if not global_episode%100:
            model_saver.save(sess, '{}/episode_{}/model'.format(model_path,global_episode))

        #play 10 games
        pyrisk_callable.play_games(params, Q_main, buffer)

        for _ in range(n_training_steps):
            batch = buffer.sample(batchsize)
            boards = np.squeeze([batch[i][0] for i in range(batchsize)])
            actions = np.array([batch[i][1] for i in range(batchsize)])
            rewards = np.array([batch[i][2] for i in range(batchsize)])
            boards_prime = np.squeeze([batch[i][3] for i in range(batchsize)])

            values = Q_target.compute_scores(boards_prime)
            max_prime = np.max(values, axis=1)
            targets = rewards + gamma*max_prime
            # train Q_main
            Q_main.train(boards, actions, targets)

        # soft update Q_target
        sess.run(update)



if __name__ == "__main__":

    # parsing argument
    parser = argparse.ArgumentParser()

    parser.add_argument("-g", "--games", type=int, default=10, help="Number of rounds to play")
    parser.add_argument("AI", help="Name of the AI class to use")
    parser.add_argument("--deal", action="store_false", default=True, help="Deal territories rather than letting players choose")
    parser.add_argument("--restart_episode", type=int, default=0, help="Number of rounds to play")

    args = parser.parse_args()
    params = {'players':['{}AI_red'.format(args.AI), '{}AI_blue'.format(args.AI), '{}AI_green'.format(args.AI)],
            'n_games': args.games, 'ai_name':args.AI, 'deal':args.deal, 'restart_episode':args.restart_episode
            }
    print(params)

    #print(params)

    main(params)
