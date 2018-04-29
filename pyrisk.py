#!/usr/bin/env python3
import numpy as np
import logging
import random
import importlib
import os
import re
import collections
import curses
from game import Game

from world import CONNECT, MAP, KEY, AREAS

LOG = logging.getLogger("pyrisk")
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--nocurses", dest="curses", action="store_false", default=True, help="Disable the ncurses map display")
parser.add_argument("--nocolor", dest="color", action="store_false", default=True, help="Display the map without colors")
parser.add_argument("-l", "--log", action="store_true", default=False, help="Write game events to a logfile")
parser.add_argument("-f", "--folder", type=str, default='logfiles', help="Folder where to store the logfile")
parser.add_argument("-d", "--delay", type=float, default=0.1, help="Delay in seconds after each action is displayed")
parser.add_argument("-s", "--seed", type=int, default=None, help="Random number generator seed")
parser.add_argument("-g", "--games", type=int, default=1, help="Number of rounds to play")
parser.add_argument("-w", "--wait", action="store_true", default=False, help="Pause and wait for a keypress after each action")
parser.add_argument("players", nargs="+", help="Names of the AI classes to use. May use 'ExampleAI*3' syntax.")
parser.add_argument("--deal", action="store_true", default=True, help="Deal territories rather than letting players choose")

args = parser.parse_args()

#NAMES = ["ALPHA", "BRAVO", "CHARLIE", "DELTA", "ECHO", "FOXTROT"]

LOG.setLevel(logging.DEBUG)
if args.log:
    os.makedirs(args.folder, exist_ok=True)
    path = '/'.join([args.folder, "pyrisk.log"])
    logging.basicConfig(filename=path, filemode="w")
elif not args.curses:
    logging.basicConfig()

if args.seed is not None:
    random.seed(args.seed)

player_classes = []
my_colors = []

for p in args.players:
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


kwargs = dict(curses=args.curses, color=args.color, delay=args.delay,
              connect=CONNECT, cmap=MAP, ckey=KEY, areas=AREAS,
              wait=args.wait, deal=args.deal)

def wrapper(stdscr, **kwargs):
    g = Game(screen=stdscr, **kwargs)
    for i, klass in enumerate(player_classes):
        g.add_player(my_colors[i], klass)
    return g.play()

if args.games == 1:
    if args.curses:
        curses.wrapper(wrapper, **kwargs)
    else:
        wrapper(None, **kwargs)
else:
    wins = collections.defaultdict(int)
    for j in range(args.games):
        kwargs['round'] = (j+1, args.games)
        kwargs['history'] = wins
        if args.curses:
            victor = curses.wrapper(wrapper, **kwargs)
        else:
            victor = wrapper(None, **kwargs)
        wins[victor] += 1
    print("Outcome of %s games" % args.games)
    for k in sorted(wins, key=lambda x: wins[x]):
        print("%s [%s]:\t%s" % (k, player_classes[my_colors.index(k)].__name__, wins[k]))
