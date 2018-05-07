from ai import AI
from display import CursesDisplay
import tensorflow as tf
import numpy as np
import pickle
import random
import collections
import curses
import time
import os
from Q_network import *
import pandas as pd




possible_attacks = ["no attack",
                    # inside North America
                    "Alaska->Northwest Territories", "Alaska->Alberta", "Alberta->Alaska",
                    "Alberta->Northwest Territories", "Alberta->Ontario", "Alberta->Western United States",
                    "Northwest Territories->Alaska","Northwest Territories->Ontario","Northwest Territories->Alberta",
                    "Northwest Territories->Greenland","Ontario->Alberta","Ontario->Greenland",
                    "Ontario->Northwest Territories","Ontario->Quebec","Ontario->Western United States",
                    "Ontario->Eastern United States",
                    "Greenland->Northwest Territories","Greenland->Ontario","Greenland->Quebec",
                    "Quebec->Eastern United States", "Quebec->Greenland","Quebec->Ontario",
                    "Western United States->Eastern United States", "Western United States->Mexico",
                    "Western United States->Alberta", "Western United States->Ontario",
                    "Mexico->Eastern United States", "Mexico->Western United States",
                    "Eastern United States->Mexico","Eastern United States->Ontario",
                    "Eastern United States->Quebec","Eastern United States->Western United States",
                    # inside South America
                    "Peru->Argentina", "Peru->Brazil", "Peru->Venezuala",
                    "Argentina->Brazil","Argentina->Peru",
                    "Brazil->Argentina","Brazil->Peru","Brazil->Venezuala",
                    "Venezuala->Brazil","Venezuala->Peru",
                    # inside Europe
                    "Scandanavia->Great Britain", "Scandanavia->Iceland",
                    "Scandanavia->Northern Europe","Scandanavia->Ukraine",
                    "Western Europe->Northern Europe","Western Europe->Southern Europe","Western Europe->Great Britain",
                    "Great Britain->Iceland", "Great Britain->Western Europe",
                    "Great Britain->Northern Europe","Great Britain->Scandanavia",
                    "Iceland->Scandanavia","Iceland->Great Britain",
                    "Southern Europe->Northern Europe","Southern Europe->Ukraine","Southern Europe->Western Europe",
                    "Ukraine->Northern Europe","Ukraine->Scandanavia","Ukraine->Southern Europe",
                    "Northern Europe->Great Britain","Northern Europe->Scandanavia",
                    "Northern Europe->Southern Europe","Northern Europe->Ukraine","Northern Europe->Western Europe",
                    # inside Africa
                    "South Africa->East Africa","South Africa->Madagascar","South Africa->Congo",
                    "Congo->North Africa","Congo->East Africa","Congo->South Africa",
                    "East Africa->Congo","East Africa->Egypt","East Africa->Madagascar",
                    "Egypt->North Africa", "Egypt->East Africa",
                    "East Africa->South Africa","East Africa->North Africa",
                    "Madagascar->East Africa","Madagascar->South Africa",
                    "North Africa->Congo","North Africa->Egypt","North Africa->East Africa",
                    # inside Asia
                    "Afghanistan->China", "Afghanistan->India", "Afghanistan->Middle East",
                    "China->Afghanistan","China->India", "China->Mongolia",
                    "China->Siberia", "China->South East Asia", "China->Ural",
                    "Siberia->China","Siberia->Irkutsk","Siberia->Mongolia",
                    "Siberia->Ural","Siberia->Yakutsk", "Afghanistan->Ural",
                    "South East Asia->China","South East Asia->India",
                    "India->Afghanistan","India->China","India->Middle East","India->South East Asia",
                    "Irkutsk->Kamchatka","Irkutsk->Mongolia","Irkutsk->Siberia","Irkutsk->Yakutsk",
                    "Japan->Kamchatka","Japan->Mongolia",
                    "Kamchatka->Irkutsk","Kamchatka->Mongolia","Kamchatka->Yakutsk","Kamchatka->Japan",
                    "Ural->Afghanistan","Ural->Siberia","Ural->China",
                    "Yakutsk->Irkutsk","Yakutsk->Kamchatka","Yakutsk->Siberia",
                    "Middle East->Afghanistan","Middle East->India",
                    "Mongolia->China","Mongolia->Irkutsk","Mongolia->Japan","Mongolia->Kamchatka","Mongolia->Siberia",
                    # inside Oceania
                    "Indonesia->Western Australia", "Indonesia->New Guinea",
                    "New Guinea->Eastern Australia","New Guinea->Indonesia","New Guinea->Western Australia",
                    "Eastern Australia->New Guinea","Eastern Australia->Western Australia",
                    "Western Australia->Eastern Australia","Western Australia->New Guinea","Western Australia->Indonesia",
                    # North america border
                    "Kamchatka->Alaska", "Alaska->Kamchatka",
                    "Greenland->Iceland","Iceland->Greenland",
                    "Mexico->Venezuala", "Venezuala->Mexico",
                    # South america border
                    "Brazil->North Africa", "North Africa->Brazil",
                    # Europe border
                    "Middle East->Southern Europe", "Southern Europe->Middle East",
                    "Middle East->Ukraine", "Ukraine->Middle East",
                    "Egypt->Southern Europe", "Southern Europe->Egypt",
                    "Southern Europe->North Africa", "North Africa->Southern Europe",
                    "Ukraine->Ural", "Ural->Ukraine",
                    "North Africa->Western Europe", "Western Europe->North Africa",
                    "Afghanistan->Ukraine","Ukraine->Afghanistan",
                    # Africa border
                    "East Africa->Middle East", "Egypt->Middle East",
                    "Middle East->Egypt", "Middle East->East Africa",
                    # Asia - Oceania
                    "Indonesia->South East Asia", "South East Asia->Indonesia"]


# initialize
board_size = 3*42 #3*len(territories) # we assume 3 players
action_size = len(possible_attacks)


class RlAI(AI):
    """
    RlAI: an Reinforcement learning AI that plays with a policy loaded
     and that save information for future training.
    """

    def start(self):
        self.turn = self.game.turn_order.index(self.player.name)
        self.board_data = []
        self.action_data = []
        self.rewards_data = []

    # random action for initial placement & reinforcement phase & freemove
    # not the main focus for now
    def initial_placement(self, empty, remaining):
        if empty:
            return random.choice(empty)
        else:
            t = random.choice(list(self.player.territories))
            return t

    def reinforce(self, available):
        border = [t for t in self.player.territories if t.border]
        result = collections.defaultdict(int)
        for i in range(available):
            t = random.choice(border)
            result[t] += 1
        return result

    def freemove(self):
        potential_src = sorted([t for t in self.player.territories if not t.border], key=lambda x: x.forces)
        if potential_src:
            border = [t for t in self.player.territories if t.border]
            if border:
                return (potential_src[-1], random.choice(border), potential_src[-1].forces - 1)
        return None

    def attack(self):
        Attacking = True
        while Attacking:
            # load board state and reshape to have this player in first position
            board = self.game.board_state
            my_board = np.reshape( np.vstack((board[self.turn:len(board)], board[:self.turn])) , (1, -1))
            controled_territories=[t.name for t in self.player.territories]

            #compute scores
            attack_scores = self.game.Q_network.compute_scores(my_board)
            attack_scores = np.squeeze(attack_scores)

            #if attack_scores[0]==0:
            #    attack_ids = np.argsort(-attack_scores)
            #else:

            # selecting the id of potential attacks with non zero scores
            potential_ids = [0]
            associated_scores = [attack_scores[0]]
            for id in range(1, action_size):
                src, dst = possible_attacks[id].split('->')
                if (src in controled_territories) and (dst not in controled_territories) and (self.world.territory(src).forces>1):
                    potential_ids.append(id)
                    associated_scores.append(attack_scores[id])

            associated_scores = np.array(associated_scores)
            sum_scores = np.sum(associated_scores)
            if sum_scores ==0:
                attack_id = np.random.choice(potential_ids)
            else:
                att_proba = np.true_divide(associated_scores, np.sum(associated_scores))
                attack_id = np.random.choice(potential_ids, p=att_proba)

            #sum_scores = sum(associated_scores)
            #att_proba = [score/sum_scores for score in associated_scores]

            """
            # rank with proba=scores
            attack_ids = np.random.choice(potential_ids, size=len(potential_ids),
                            replace=False, p=associated_scores)
            """
            # choose with proba=scores
            #attack_id = np.random.choice(potential_ids, p=att_proba)
            self.board_data.append(my_board)
            self.action_data.append(attack_id)
            if attack_id == 0: # it means no attack
                Attacking = False
                return None
            else:
                src, dst = possible_attacks[attack_id].split('->')
                yield (src, dst, None, None) # full attack for now, we will see in the future if we can choose a more advanced strategy


            """
            # when trained
            attack_ids = np.argsort(-attack_scores)]
            """

            """
            # then we need to choose a valid attack
            for i in range(action_size):
                #attack_id = np.random.choice(action_size, p=attack_scores)
                attack_id = attack_ids[i]

                if attack_id == 0: # it means no attack
                    self.board_data.append(my_board)
                    self.action_data.append(attack_id)
                    Attacking = False
                    return None
                else:
                    src, dst = possible_attacks[attack_id].split('->')
                    if (src in controled_territories) and (dst not in controled_territories) and (self.world.territory(src).forces>1):
                        break

            self.board_data.append(my_board)
            self.action_data.append(attack_id)
            yield (src, dst, None, None) # full attack for now, we will see in the future if we can choose a more advanced strategy
            """
    # to give rewards
    def event(self, msg):
        # strategy 1 : reward at the end
        statement = msg[0]
        if statement == "victory":
            winner = msg[1].name
            N = len(self.action_data)

            self.rewards_data = [0]*(N-1)
            if self.player.name == winner:
                self.rewards_data.append(100)
            else:
                self.rewards_data.append(-20)


            # transfer directly to buffer:
            for i in range(N-1):
                self.game.buffer.append((self.board_data[i], self.action_data[i],
                        self.rewards_data[i], self.board_data[i+1]))
            self.game.buffer.append((self.board_data[N-1], self.action_data[N-1],
                    self.rewards_data[N-1], self.board_data[N-1]))
