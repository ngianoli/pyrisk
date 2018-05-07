#from display import Display, CursesDisplay
from player import Player
from territory import World
from world import CONNECT, AREAS, MAP, KEY
import logging
LOG = logging.getLogger("pyrisk")
import random
import numpy as np



territories = ["Alaska","Northwest Territories","Greenland","Alberta","Ontario","Quebec","Western United States","Eastern United States","Mexico",
               "Venezuala","Peru","Argentina","Brazil",
               "Iceland","Great Britain","Scandanavia","Western Europe","Northern Europe","Southern Europe","Ukraine",
               "North Africa","Egypt","East Africa","Congo","South Africa","Madagascar",
               "Middle East","Ural","Siberia","Yakutsk","Irkutsk","Kamchatka","Afghanistan","Mongolia","China","Japan","India","South East Asia",
               "Indonesia","New Guinea","Western Australia", "Eastern Australia"]

possible_attacks = ["no attack",
                    # inside North America
                    "T;Alaska, T;Northwest Territories", "T;Alaska, T;Alberta", "T;Alberta, T;Alaska",
                    "T;Alberta, T;Northwest Territories", "T;Alberta, T;Ontario", "T;Alberta, T;Western United States",
                    "T;Northwest Territories, T;Alaska","T;Northwest Territories, T;Ontario","T;Northwest Territories, T;Alberta",
                    "T;Northwest Territories, T;Greenland","T;Ontario, T;Alberta","T;Ontario, T;Greenland",
                    "T;Ontario, T;Northwest Territories","T;Ontario, T;Quebec","T;Ontario, T;Western United States",
                    "T;Ontario, T;Eastern United States",
                    "T;Greenland, T;Northwest Territories","T;Greenland, T;Ontario","T;Greenland, T;Quebec",
                    "T;Quebec, T;Eastern United States", "T;Quebec, T;Greenland","T;Quebec, T;Ontario",
                    "T;Western United States, T;Eastern United States", "T;Western United States, T;Mexico",
                    "T;Western United States, T;Alberta", "T;Western United States, T;Ontario",
                    "T;Mexico, T;Eastern United States", "T;Mexico, T;Western United States",
                    "T;Eastern United States, T;Mexico","T;Eastern United States, T;Ontario",
                    "T;Eastern United States, T;Quebec","T;Eastern United States, T;Western United States",
                    # inside South America
                    "T;Peru, T;Argentina", "T;Peru, T;Brazil", "T;Peru, T;Venezuala",
                    "T;Argentina, T;Brazil","T;Argentina, T;Peru",
                    "T;Brazil, T;Argentina","T;Brazil, T;Peru","T;Brazil, T;Venezuala",
                    "T;Venezuala, T;Brazil","T;Venezuala, T;Peru",
                    # inside Europe
                    "T;Scandanavia, T;Great Britain", "T;Scandanavia, T;Iceland",
                    "T;Scandanavia, T;Northern Europe","T;Scandanavia, T;Ukraine",
                    "T;Western Europe, T;Northern Europe","T;Western Europe, T;Southern Europe","T;Western Europe, T;Great Britain",
                    "T;Great Britain, T;Iceland", "T;Great Britain, T;Western Europe",
                    "T;Great Britain, T;Northern Europe","T;Great Britain, T;Scandanavia",
                    "T;Iceland, T;Scandanavia","T;Iceland, T;Great Britain",
                    "T;Southern Europe, T;Northern Europe","T;Southern Europe, T;Ukraine","T;Southern Europe, T;Western Europe",
                    "T;Ukraine, T;Northern Europe","T;Ukraine, T;Scandanavia","T;Ukraine, T;Southern Europe",
                    "T;Northern Europe, T;Great Britain","T;Northern Europe, T;Scandanavia",
                    "T;Northern Europe, T;Southern Europe","T;Northern Europe, T;Ukraine","T;Northern Europe, T;Western Europe",
                    # inside Africa
                    "T;South Africa, T;East Africa","T;South Africa, T;Madagascar","T;South Africa, T;Congo",
                    "T;Congo, T;North Africa","T;Congo, T;East Africa","T;Congo, T;South Africa",
                    "T;East Africa, T;Congo","T;East Africa, T;Egypt","T;East Africa, T;Madagascar",
                    "T;Egypt, T;North Africa", "T;Egypt, T;East Africa",
                    "T;East Africa, T;South Africa","T;East Africa, T;North Africa",
                    "T;Madagascar, T;East Africa","T;Madagascar, T;South Africa",
                    "T;North Africa, T;Congo","T;North Africa, T;Egypt","T;North Africa, T;East Africa",
                    # inside Asia
                    "T;Afghanistan, T;China", "T;Afghanistan, T;India", "T;Afghanistan, T;Middle East",
                    "T;China, T;Afghanistan","T;China, T;India", "T;China, T;Mongolia",
                    "T;China, T;Siberia", "T;China, T;South East Asia", "T;China, T;Ural",
                    "T;Siberia, T;China","T;Siberia, T;Irkutsk","T;Siberia, T;Mongolia",
                    "T;Siberia, T;Ural","T;Siberia, T;Yakutsk", "T;Afghanistan, T;Ural",
                    "T;South East Asia, T;China","T;South East Asia, T;India",
                    "T;India, T;Afghanistan","T;India, T;China","T;India, T;Middle East","T;India, T;South East Asia",
                    "T;Irkutsk, T;Kamchatka","T;Irkutsk, T;Mongolia","T;Irkutsk, T;Siberia","T;Irkutsk, T;Yakutsk",
                    "T;Japan, T;Kamchatka","T;Japan, T;Mongolia",
                    "T;Kamchatka, T;Irkutsk","T;Kamchatka, T;Mongolia","T;Kamchatka, T;Yakutsk","T;Kamchatka, T;Japan",
                    "T;Ural, T;Afghanistan","T;Ural, T;Siberia","T;Ural, T;China",
                    "T;Yakutsk, T;Irkutsk","T;Yakutsk, T;Kamchatka","T;Yakutsk, T;Siberia",
                    "T;Middle East, T;Afghanistan","T;Middle East, T;India",
                    "T;Mongolia, T;China","T;Mongolia, T;Irkutsk","T;Mongolia, T;Japan","T;Mongolia, T;Kamchatka","T;Mongolia, T;Siberia",
                    # inside Oceania
                    "T;Indonesia, T;Western Australia", "T;Indonesia, T;New Guinea",
                    "T;New Guinea, T;Eastern Australia","T;New Guinea, T;Indonesia","T;New Guinea, T;Western Australia",
                    "T;Eastern Australia, T;New Guinea","T;Eastern Australia, T;Western Australia",
                    "T;Western Australia, T;Eastern Australia","T;Western Australia, T;New Guinea","T;Western Australia, T;Indonesia",
                    # North america border
                    "T;Kamchatka, T;Alaska", "T;Alaska, T;Kamchatka",
                    "T;Greenland, T;Iceland","T;Iceland, T;Greenland",
                    "T;Mexico, T;Venezuala", "T;Venezuala, T;Mexico",
                    # South america border
                    "T;Brazil, T;North Africa", "T;North Africa, T;Brazil",
                    # Europe border
                    "T;Middle East, T;Southern Europe", "T;Southern Europe, T;Middle East",
                    "T;Middle East, T;Ukraine", "T;Ukraine, T;Middle East",
                    "T;Egypt, T;Southern Europe", "T;Southern Europe, T;Egypt",
                    "T;Southern Europe, T;North Africa", "T;North Africa, T;Southern Europe",
                    "T;Ukraine, T;Ural", "T;Ural, T;Ukraine",
                    "T;North Africa, T;Western Europe", "T;Western Europe, T;North Africa",
                    "T;Afghanistan, T;Ukraine","T;Ukraine, T;Afghanistan",
                    # Africa border
                    "T;East Africa, T;Middle East", "T;Egypt, T;Middle East",
                    "T;Middle East, T;Egypt", "T;Middle East, T;East Africa",
                    # Asia - Oceania
                    "T;Indonesia, T;South East Asia", "T;South East Asia, T;Indonesia"]




class Game(object):
    """
    This class represents an individual game, and contains the main game logic.
    """
    defaults = {
        "connect": CONNECT, #the territory connection graph (see world.py)
        "areas": AREAS, #the territory->continent mapping, and values
        "cmap": MAP, #the ASCII art map to use
        "ckey": KEY, #the territority->char mapping key for the map
        "round": None, #the round number
        "history": {}, #the win/loss history for each player, for multiple rounds
        "deal": True #deal out territories rather than let players choose
    }

    def __init__(self, Q_network=None, buffer=None, **options):
        self.options = self.defaults.copy()
        self.options.update(options)

        self.world = World()
        self.world.load(self.options['areas'], self.options['connect'])

        self.board_state = None
        self.players = {}
        self.player_to_id={}

        self.turn = 0
        self.turn_order = []

        # sot that AI can access the same DNN
        self.Q_network=Q_network
        self.buffer = buffer

    def add_player(self, name, ai_class, **ai_kwargs):
        assert name not in self.players
        color_to_int={'red':1, 'green':2, 'yellow':3, 'blue':4, 'purple':5}
        color=color_to_int[name]
        player = Player(name, color, self, ai_class, ai_kwargs)
        self.players[name] = player

    @property
    def player(self):
        """Property that returns the correct player object for this turn."""
        return self.players[self.turn_order[self.turn % len(self.players)]]

    def aiwarn(self, *args):
        """Generate a warning message when an AI player tries to break the rules."""
        logging.getLogger("pyrisk.player.%s" % self.player.ai.__class__.__name__).warn(*args)

    def event(self, msg, territory=None, player=None):
        """
        Record any game action.
        `msg` is a tuple describing what happened.
        `territory` is a list of territory objects to be highlighted, if any
        `player` is a list of player names to be highlighted, if any

        Calling this method triggers the display to be updated, and any AI
        players that have implemented event() to be notified.
        """
        #self.display.update(msg, territory=territory, player=player)

        LOG.info([str(m) for m in msg])
        for p in self.players.values():
            p.ai.event(msg)

    def play(self):
        assert 2 <= len(self.players) <= 5
        self.turn_order = list(self.players)
        random.shuffle(self.turn_order)

        for i, player in enumerate(self.turn_order):
            self.player_to_id.update({player:i})

        for i, name in enumerate(self.turn_order):
            #self.players[name].color = i + 1
            #self.players[name].ord = ord('\/-|+*'[i])   ## bullshit
            self.players[name].ai.start()

        self.board_state = np.zeros((len(self.players), 42)) #42 territories

        self.event(("start", ))
        live_players = len(self.players)
        self.initial_placement()
        stalemate=False
        while live_players > 1:
            if self.player.alive:
                choices = self.player.ai.reinforce(self.player.reinforcements)
                assert sum(choices.values()) == self.player.reinforcements
                for tt, ff in choices.items():
                    t = self.world.territory(tt)
                    f = int(ff)
                    if t is None:
                        self.aiwarn("reinforce invalid territory %s", tt)
                        continue
                    if t.owner != self.player:
                        self.aiwarn("reinforce unowned territory %s", t.name)
                        continue
                    if f < 0:
                        self.aiwarn("reinforce invalid count %s", f)
                        continue
                    t.forces += f
                    self.event(("reinforce", self.player, t, f), territory=[t], player=[self.player.name])
                    #update board state
                    player_id = self.player_to_id[self.player.name]
                    territory_id = territories.index(t.name)
                    self.board_state[player_id, territory_id]+=f

                for src, target, attack, move in self.player.ai.attack():
                    st = self.world.territory(src)
                    tt = self.world.territory(target)
                    if st is None:
                        self.aiwarn("attack invalid src %s", src)
                        continue
                    if tt is None:
                        self.aiwarn("attack invalid target %s", target)
                        continue
                    if st.owner != self.player:
                        self.aiwarn("attack unowned src %s", st.name)
                        continue
                    if tt.owner == self.player:
                        self.aiwarn("attack owned target %s", tt.name)
                        continue
                    if tt not in st.connect:
                        self.aiwarn("attack unconnected %s %s", st.name, tt.name)
                        continue
                    initial_forces = (st.forces, tt.forces)
                    opponent = tt.owner
                    victory = self.combat(st, tt, attack, move)
                    final_forces = (st.forces, tt.forces)
                    self.event(("conquer" if victory else "defeat", self.player, opponent, st, tt, initial_forces, final_forces), territory=[st, tt], player=[self.player.name, tt.owner.name])
                    attacker_id = self.player_to_id[self.player.name]
                    defender_id = self.player_to_id[opponent.name]
                    att_territory_id = territories.index(st.name)
                    def_territory_id = territories.index(tt.name)
                    if victory:
                        self.board_state[attacker_id, att_territory_id]=final_forces[0]
                        self.board_state[attacker_id, def_territory_id]=final_forces[1]
                        self.board_state[defender_id, def_territory_id]=0
                    else:
                        self.board_state[attacker_id, att_territory_id]=final_forces[0]
                        self.board_state[defender_id, def_territory_id]=final_forces[1]


                freemove = self.player.ai.freemove()
                if freemove:
                    src, target, count = freemove
                    st = self.world.territory(src)
                    tt = self.world.territory(target)
                    f = int(count)
                    valid = True
                    if st is None:
                        self.aiwarn("freemove invalid src %s", src)
                        valid = False
                    if tt is None:
                        self.aiwarn("freemove invalid target %s", target)
                        valid = False
                    if st.owner != self.player:
                        self.aiwarn("freemove unowned src %s", st.name)
                        valid = False
                    if tt.owner != self.player:
                        self.aiwarn("freemove unowned target %s", tt.name)
                        valid = False
                    if not 0 <= f < st.forces:
                        self.aiwarn("freemove invalid count %s", f)
                        valid = False
                    if valid:
                        st.forces -= count
                        tt.forces += count
                        self.event(("move", self.player, st, tt, count), territory=[st, tt], player=[self.player.name])
                        # update board
                        player_id = self.player_to_id[self.player.name]
                        src_territory_id = territories.index(st.name)
                        dest_territory_id = territories.index(tt.name)
                        self.board_state[player_id, src_territory_id]-=count
                        self.board_state[player_id, dest_territory_id]+=count

                live_players = len([p for p in self.players.values() if p.alive])
            self.turn += 1
            if self.turn>1000:
                stalemate=True
                break
        winner = [p for p in self.players.values() if p.alive][0]
        #if stalemate:
        # winner.name='Stale Horse'
        self.event(("victory", winner), player=[self.player.name])
        for p in self.players.values():
            p.ai.end()
        return winner.name

    def combat(self, src, target, f_atk, f_move):
        n_atk = src.forces
        n_def = target.forces

        if f_atk is None:
            f_atk = lambda a, d: True
        if f_move is None:
            f_move = lambda a: a - 1

        while n_atk > 1 and n_def > 0 and f_atk(n_atk, n_def):
            atk_dice = min(n_atk - 1, 3)
            atk_roll = sorted([random.randint(1, 6) for i in range(atk_dice)], reverse=True)
            def_dice = min(n_def, 2)
            def_roll = sorted([random.randint(1, 6) for i in range(def_dice)], reverse=True)

            for a, d in zip(atk_roll, def_roll):
                if a > d:
                    n_def -= 1
                else:
                    n_atk -= 1

        if n_def == 0:
            move = f_move(n_atk)
            min_move = min(n_atk - 1, 3)
            max_move = n_atk - 1
            if move < min_move:
                self.aiwarn("combat invalid move request %s (%s-%s)", move, min_move, max_move)
                move = min_move
            if move > max_move:
                self.aiwarn("combat invalid move request %s (%s-%s)", move, min_move, max_move)
                move = max_move
            src.forces = n_atk - move
            target.forces = move
            target.owner = src.owner
            return True
        else:
            src.forces = n_atk
            target.forces = n_def
            return False

    def initial_placement(self):
        empty = list(self.world.territories.values())
        available = 35 - 2*len(self.players)
        remaining = {p: available for p in self.players}

        if self.options['deal']:
            random.shuffle(empty)
            while empty:
                t = empty.pop()
                t.forces += 1
                remaining[self.player.name] -= 1
                t.owner = self.player
                self.event(("deal", self.player, t), territory=[t], player=[self.player.name])
                # update board
                player_id = self.player_to_id[self.player.name]
                territory_id = territories.index(t.name)
                self.board_state[player_id, territory_id]+=1
                self.turn += 1
        else:
            while empty:
                choice = self.player.ai.initial_placement(empty, remaining[self.player.name])
                t = self.world.territory(choice)
                if t is None:
                    self.aiwarn("invalid territory choice %s", choice)
                    self.turn += 1
                    continue
                if t not in empty:
                    self.aiwarn("initial invalid empty territory %s", t.name)
                    self.turn += 1
                    continue
                t.forces += 1
                t.owner = self.player
                remaining[self.player.name] -= 1
                empty.remove(t)
                self.event(("claim", self.player, t), territory=[t], player=[self.player.name])
                # update board
                player_id = self.player_to_id[self.player.name]
                territory_id = territories.index(t.name)
                self.board_state[player_id, territory_id]+=1

                self.turn += 1

        while sum(remaining.values()) > 0:
            if remaining[self.player.name] > 0:
                choice = self.player.ai.initial_placement(None, remaining[self.player.name])
                t = self.world.territory(choice)
                if t is None:
                    self.aiwarn("initial invalid territory %s", choice)
                    self.turn += 1
                    continue
                if t.owner != self.player:
                    self.aiwarn("initial unowned territory %s", t.name)
                    self.turn += 1
                    continue
                t.forces += 1
                remaining[self.player.name] -= 1
                self.event(("reinforce", self.player, t, 1), territory=[t], player=[self.player.name])
                # update board
                player_id = self.player_to_id[self.player.name]
                territory_id = territories.index(t.name)
                self.board_state[player_id,territory_id]+=1
                self.turn += 1
