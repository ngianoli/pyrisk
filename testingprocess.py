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
    tested_episode=params['tested_episode']
    model_path = 'model_files/{}/episode_{}/model'.format(NAME,tested_episode)

    scores_folder = 'score_files/{}/episode_{}'.format(NAME,tested_episode)
    os.makedirs(score_files, exist_ok=True)

    # initialize
    board_size = 3*42 #3*len(territories) # we assume 3 players
    action_size = 167 #len(possible_attacks)
    learning_rate =0.001
    gamma = .95  # discount
    max_length=500000 # for the buffer
    n_training_steps = 10
    batchsize = 32

    max_length = 100000000


    buffer = ReplayBuffer(max_length)
    sess = tf.Session()

    loader = tf.train.import_meta_graph('{}.meta'.format(model_path))
    #'/Users/ngianoli/Desktop/Risk_project/pyrisk/model_files/Gather/episode_2/model.meta')
    loader.restore(sess, model_path)
    graph = tf.get_default_graph()
    graph.get_collection('trainable_variables')

    loaded_vars = {}
    for var in graph.get_collection('trainable_variables'):
        if var.name[:4]=='main':
            ref = var.name.split('/')[1][:-7])
            loaded_vars.update({ref:var})

    Q_main = Q_network(board_size, action_size, sess, variables=loaded_vars)


    all_results={}

    # game against 1 Stupid
    params.update({'players':['{}AI_blue'.format(args.AI), 'StupidAI_green'.format(args.AI), '{}AI_red'.format(args.AI)]})
    wins = pyrisk_callable.play_games(params, Q_main, buffer)
    all_results.update('1_Stupid' : wins)
    buffer = ReplayBuffer(max_length) #restart buffer

    # game against 2 Stupid
    params.update({'players':['StupidAI_blue'.format(args.AI), 'StupidAI_green'.format(args.AI), '{}AI_red'.format(args.AI)]})
    wins = pyrisk_callable.play_games(params, Q_main, buffer)
    all_results.update('2_Stupid' : wins)
    buffer = ReplayBuffer(max_length) #restart buffer

    # game against 1 Better
    params.update({'players':['{}AI_blue'.format(args.AI), 'BetterAI_green'.format(args.AI), '{}AI_red'.format(args.AI)]})
    wins = pyrisk_callable.play_games(params, Q_main, buffer)
    all_results.update('1_Better' : wins)
    buffer = ReplayBuffer(max_length) #restart buffer

    # game against 2 Better
    params.update({'players':['BetterAI_blue'.format(args.AI), 'BetterAI_green'.format(args.AI), '{}AI_red'.format(args.AI)]})
    wins = pyrisk_callable.play_games(params, Q_main, buffer)
    all_results.update('2_Better' : wins)
    buffer = ReplayBuffer(max_length) #restart buffer

    # game against 1 Clever
    params.update({'players':['{}AI_blue'.format(args.AI), 'CleverAI_green'.format(args.AI), '{}AI_red'.format(args.AI)]})
    wins = pyrisk_callable.play_games(params, Q_main, buffer)
    all_results.update('1_Clever' : wins)
    buffer = ReplayBuffer(max_length) #restart buffer

    # game against 2 Clever
    params.update({'players':['CleverAI_blue'.format(args.AI), 'CleverAI_green'.format(args.AI), '{}AI_red'.format(args.AI)]})
    wins = pyrisk_callable.play_games(params, Q_main, buffer)
    all_results.update('2_Clever' : wins)
    buffer = ReplayBuffer(max_length) #restart buffer





    """
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

    """

if __name__ == "__main__":

    # parsing argument
    parser = argparse.ArgumentParser()

    parser.add_argument("-g", "--games", type=int, default=1, help="Number of rounds to play")
    parser.add_argument("AI", help="Name of the AI class to use")
    parser.add_argument("--deal", action="store_false", default=True, help="Deal territories rather than letting players choose")
    parser.add_argument("--tested_episode", type=int, default=0, help="Number of rounds to play")

    args = parser.parse_args()
    params = {'n_games': args.games, 'ai_name':args.AI, 'deal':args.deal,
                'tested_episode':args.tested_episode}

    #print(params)

    main(params)
