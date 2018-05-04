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
    max_length=500000 # for the buffer
    n_training_steps = 10
    batchsize = 32


    buffer = ReplayBuffer(max_length)
    sess = tf.Session()
    # optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate)
    # initialize network
    with tf.variable_scope("main"):
        Q_main = Q_network(board_size, action_size, sess, optimizer, 'main')
    with tf.variable_scope("target"):
        Q_target = Q_network(board_size, action_size, sess, optimizer, 'target')
    # build ops
    update = build_target_update("main", "target")  # call sess.run(update) to copy from principal to target

    model_saver = tf.train.Saver(max_to_keep=1)
    model_path = 'model_files/{}'.format(NAME)
    os.makedirs(model_path, exist_ok=True)

    """
    # init variables
    model_list = []
    for file in os.listdir(model_path):
        if len(file)>7:
            if file[:7]=='episode':
                model_list.append(int(file.split('_')[1]))
    """

    if global_episode==0:
        sess.run(tf.global_variables_initializer())
        #save model
        model_saver.save(sess, '{}/episode_{}/model'.format(model_path,global_episode))
    elif global_episode==2:
        model_saver.restore(sess, '{}/episode_{}/model'.format(model_path,0))
        model_saver.save(sess, '{}/episode_{}/model'.format(model_path,global_episode))


    else :
        #save model
        Q_main.restore('{}episode_{}/main'.format(model_path,global_episode))
        Q_target.restore('{}episode_{}/target'.format(model_path,global_episode))



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
        #subprocess.call('python pyrisk.py {}AI_blue {}AI_red {}AI_green -g 2 --nocurse -l'.format(NAME, NAME, NAME)) # can be parallelized
        """
        data_to_gather =[]
        for player in ['red', 'blue', 'green']:
            folder = 'game_files/{}'.format(player)
            for file in os.listdir(folder):
                if file[:4]=='game':
                    data_to_gather.append('{}/{}'.format(folder,file))

        for path in data_to_gather:
            buffer.append_frame(pd.read_csv(path, sep=',', header=None))
            os.remove(path)
        """

        for _ in range(n_training_steps):
            batch = buffer.sample(batchsize)
            boards = np.squeeze([batch[i][0] for i in range(batchsize)])
            actions = np.array([batch[i][1] for i in range(batchsize)])
            rewards = np.array([batch[i][2] for i in range(batchsize)])
            boards_prime = np.squeeze([batch[i][3] for i in range(batchsize)])
            """
            except ValueError:
                print('###########################################')
                print('###########################################')
                print('###########################################')
                print (data_to_gather)
                print('###########################################')
                print('###########################################')
                print('###########################################')
            """
            values = Q_target.compute_scores(boards_prime)
            max_prime = np.max(values, axis=1)
            targets = rewards + gamma*max_prime
            # train Q_main
            Q_main.train(boards, actions, targets)

        """
        #save Q_main for agents
        Q_main.save(main_model_path)
        """

        # soft update Q_target
        sess.run(update)








if __name__ == "__main__":

    # parsing argument
    parser = argparse.ArgumentParser()

    #parser.add_argument("-l", "--log", action="store_true", default=False, help="Write game events to a logfile")
    #parser.add_argument("-f", "--folder", type=str, default='logfiles', help="Folder where to store the logfile")
    #parser.add_argument("-d", "--delay", type=float, default=0.1, help="Delay in seconds after each action is displayed")
    #parser.add_argument("-s", "--seed", type=int, default=None, help="Random number generator seed")
    parser.add_argument("-g", "--games", type=int, default=1, help="Number of rounds to play")
    #parser.add_argument("-w", "--wait", action="store_true", default=False, help="Pause and wait for a keypress after each action")
    #parser.add_argument("AI", nargs="+", help="Name of the AI class to use")
    parser.add_argument("AI", help="Name of the AI class to use")
    parser.add_argument("--deal", action="store_false", default=True, help="Deal territories rather than letting players choose")
    parser.add_argument("--restart_episode", type=int, default=0, help="Number of rounds to play")

    args = parser.parse_args()
    params = {'players':['{}AI_blue'.format(args.AI), '{}AI_green'.format(args.AI), '{}AI_red'.format(args.AI)],
            'n_games': args.games, 'ai_name':args.AI, 'deal':args.deal, 'restart_episode':args.restart_episode
            }

    #print(params)

    main(params)
