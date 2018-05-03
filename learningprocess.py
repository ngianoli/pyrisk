from ai import AI
from display import CursesDisplay
import tensorflow as tf
import numpy as np
import pickle
import random
import collections
import curses
import time
from Q_network import *
import os
import sys

def main(NAME):
    # initialize
    board_size = 3*42 #3*len(territories) # we assume 3 players
    action_size = 167 #len(possible_attacks)
    learning_rate =0.001
    max_length=500000 # for the buffer
    n_training_steps = 10
    batchsize = 32


    buffer = ReplayBuffer(max_length)
    sess = tf.Session()
    # optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate)
    # initialize network
    with tf.variable_scope("main"):
        Q_main = Q_network(board_size, action_size, sess, optimizer, name='main')
    with tf.variable_scope("target"):
        Q_target = Q_network(board_size, action_size, sess, optimizer, name='target')
    # build ops
    update = build_target_update("main", "target")  # call sess.run(update) to copy from principal to target

    #model_saver = tf.train.Saver()
    model_path = 'model_files/{}/'.format(NAME)
    main_model_path = 'model_files/{}/current/'.format(NAME)

    os.makedirs(model_path, exist_ok=True)
    os.makedirs(main_model_path, exist_ok=True)

    # init variables
    model_list = []
    for file in os.listdir(model_path):
        if len(file)>7:
            if file[:7]=='episode':
                model_list.append(int(file.split('_')[1]))

    if len(model_list)==0:
        sess.run(tf.global_variables_initializer())
        global_episode=0
        #save model
        Q_main.save(save_path='{}episode_{}/main'.format(model_path,global_episode))
        Q_target.save(save_path='{}episode_{}/target'.format(model_path,global_episode))
        #save main for agent
        Q_main.save(save_path=main_model_path)
    else:
        global_episode = max(model_list)
        #save model
        Q_main.restore('{}episode_{}/main'.format(model_path,global_episode))
        Q_target.restore('{}episode_{}/target'.format(model_path,global_episode))
        #save main for agent
        Q_main.save(save_path=main_model_path)

    sess.run(update)


    """
    n_training_steps=10
    n_episodes = 5000
    tau = 10
    """
    n_training_steps=1
    n_episodes = 5
    tau = 10 # soft update of Q_target

    for _ in range(n_episodes):
        global_episode+=1

        # save model every 100 episodes = every 1000 games
        if not global_episode%100:
            Q_main.save(save_path='{}episode_{}/main'.format(model_path,global_episode))
            Q_target.save(save_path='{}episode_{}/target'.format(model_path,global_episode))

        #play 10 games
        os.system('python pyrisk.py {}AI_blue {}AI_red {}AI_green -g 2 --nocurse -l'.format(NAME, NAME, NAME)) # can be parallelized

        data_to_gather =[]
        for player in ['red', 'blue', 'green']:
            folder = 'game_files/{}'.format(player)
            for file in os.listdir(folder):
                if file[:4]=='game':
                    data_to_gather.append('{}/{}'.format(folder,file))

        for path in data_to_gather:
            buffer.append_frame(pd.read_csv(path, sep=',', header=None))
            os.remove(path)

        for _ in range(n_training_steps):
            batch = buffer.sample(batchsize)
            boards = np.array([batch[i][0] for i in range(batchsize)])
            actions = np.array([batch[i][1] for i in range(batchsize)])
            rewards = np.array([batch[i][2] for i in range(batchsize)])
            boards_prime = np.array([batch[i][3] for i in range(batchsize)])
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
            Q_main.train(states, actions, targets)

        #save Q_main for agents
        Q_main.save(save_path=main_model_path)

        # update Q_target
        sess.run(update)










    """

    for episode in range(episodes):

        d = False
        reward_sum = 0
        step = 0
        #evaluation
        if not episode%100:
            score = eval_score()
            print('Episode {} -> Score : {}'.format(episode,score))
            if score > 499:
                break # stop learning

        while not d and step < episode_max_length:
            #get batch
            batch = buffer.sample(batchsize)
            states = np.array([batch[i][0] for i in range(batchsize)])
            actions = np.array([batch[i][1] for i in range(batchsize)])
            rewards = np.array([batch[i][2] for i in range(batchsize)])
            states_prime = np.array([batch[i][3] for i in range(batchsize)])

            values = Qtarget.compute_Qvalues(states_prime)
            max_prime = np.max(values, axis=1)
            targets = rewards + gamma*max_prime
            # train Qprincipal
            Qprincipal.train(states, actions, targets)

            # select new action
            if epsilon > 0.1:
                epsilon*=epsilon_weakening
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                values = Qprincipal.compute_Qvalues(np.expand_dims(obs,0))
                action = np.argmax(values.flatten())

            new_obs, r, d, _ = env.step(action)

            if d :
                buffer.append((obs, action, -50, new_obs))
            else :
                buffer.append((obs, action, r, new_obs))

            obs = new_obs
            reward_sum += r
            step += 1
            counter += 1

            buffer.pop()

            #Update target network
            if not counter%tau:
                sess.run(update)
    """







if __name__ == "__main__":
    main(sys.argv[1])
