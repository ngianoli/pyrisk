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
action_space = len(possible_attacks)
learning_rate =0.001

load_session = False
sess = tf.Session()


# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate)

# initialize network
with tf.variable_scope("principal"):
    Q_net = Q_network(board_size, action_space, sess, optimizer)



if load_session == True:
    saver.restore(sess, "/Users/ngianoli/Desktop/Risk_project/model.ckpt")
else:
    sess.run(tf.global_variables_initializer())
#buffer = ReplayBuffer(maxlength)
saver = tf.train.Saver()


action_data, board_data, rewards_data = [], [], []



class GatherAI(AI):
    """
    GatherAI: an AI that plays with a policy loaded, and that save information for future training.
    """

    def start(self):
        self.turn = self.game.turn_order.index(self.player.name)

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
            #compute scores
            attack_scores = Q_net.compute_scores(my_board)
            attack_scores = np.squeeze(attack_scores)

            controled_territories=[t.name for t in self.player.territories]

            valid_att = False
            while not valid_att:
                attack_id = np.random.choice(action_space, p=attack_scores)
                """
                # when trained
                attack_id = np.argmax(final_probs)
                """
                if attack_id == 0:
                    board_data.append(my_board)
                    action_data.append(attack_id)
                    valid_att = True
                    Attacking = False
                    return None
                else:
                    src, dst = possible_attacks[attack_id].split('->')
                    if (src in controled_territories) and (dst not in controled_territories) and (self.world.territory(src).forces>1):
                        valid_att = True
                        board_data.append(my_board)
                        action_data.append(attack_id)
                        yield (src, dst, None, None) # full attack for now, we will see in the future if we can choose a more advanced strategy
