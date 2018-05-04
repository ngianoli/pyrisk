#!/usr/bin/env python3
import numpy as np
import logging
import random
import importlib
import os
import re
import collections
import curses
from game_bis import Game
import argparse

from world import CONNECT, MAP, KEY, AREAS

def play_games(params, Q_network, buffer):
    # params contain n_games, players_input, deal
    LOG = logging.getLogger("pyrisk")


    kwargs = dict(connect=CONNECT, cmap=MAP, ckey=KEY, areas=AREAS, deal=params['deal'])



    #parser = argparse.ArgumentParser()
    #parser.add_argument("-l", "--log", action="store_true", default=False, help="Write game events to a logfile")
    #parser.add_argument("-f", "--folder", type=str, default='logfiles', help="Folder where to store the logfile")
    #parser.add_argument("-d", "--delay", type=float, default=0.1, help="Delay in seconds after each action is displayed")
    #parser.add_argument("-s", "--seed", type=int, default=None, help="Random number generator seed")
    #parser.add_argument("-g", "--games", type=int, default=1, help="Number of rounds to play")
    #parser.add_argument("players", nargs="+", help="Names of the AI classes to use. May use 'ExampleAI*3' syntax.")
    #parser.add_argument("--deal", action="store_true", default=True, help="Deal territories rather than letting players choose")

    #args = parser.parse_args()



    LOG.setLevel(logging.DEBUG)
    arglog=False
    args_curses=False
    if arglog:
        os.makedirs(args.folder, exist_ok=True)
        path = '/'.join([args.folder, "pyrisk.log"])
        logging.basicConfig(filename=path, filemode="w")

    #elif not args_curses:
    #    logging.basicConfig()

    args_seed = None
    if args_seed is not None:
        random.seed(args_seed)

    player_classes = []
    my_colors = []

    for p in params['players']:
        match = re.match(r"(\w+)?", p)
        if match:
            #import mechanism
            #we expect a particular filename->classname mapping such that
            #ExampleAI resides in ai/example.py, FooAI in ai/foo.py etc.
            name, color = match.group(1).split('_')
            package = name[:-2].lower()
            if color in ['red', 'green', 'blue', 'yellow', 'purple']:
                my_colors.append(color)
            else:
                raise Exception("Color not recognized : {}".format(color))

            try:
                klass = getattr(importlib.import_module("ai."+package), name)
                player_classes.append(klass)
            except:
                print("Unable to import AI %s from ai/%s.py" % (name, package))
                raise

    """kwargs = dict(curses=args.curses, color=args.color, delay=args.delay,
                  connect=CONNECT, cmap=MAP, ckey=KEY, areas=AREAS,
                  wait=args.wait, deal=args.deal)"""

    def wrapper(stdscr, **kwargs):
        g = Game(screen=stdscr, Q_network=Q_network, buffer=buffer, **kwargs)
        for i, klass in enumerate(player_classes):
            g.add_player(my_colors[i], klass)
        return g.play()

    if params['n_games'] == 1:
        wrapper(None, **kwargs)
    else:
        wins = collections.defaultdict(int)
        for j in range(params['n_games']):
            kwargs['round'] = (j+1, params['n_games'])
            kwargs['history'] = wins
            victor = wrapper(None, **kwargs)
            wins[victor] += 1
        print("Outcome of %s games" % params['n_games'])
        for k in sorted(wins, key=lambda x: wins[x]):
            print("%s [%s]:\t%s" % (k, player_classes[my_colors.index(k)].__name__, wins[k]))
