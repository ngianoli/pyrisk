from ai import AI
import random
import collections

class PeacefulAI(AI):
    """
    PeacefulAI: Same as BetterAI, but never attacks.
    """
    def start(self):
        self.area_priority = list(self.world.areas)
        random.shuffle(self.area_priority)

    def priority(self):
        priority = sorted([t for t in self.player.territories if t.border],
                          key=lambda x: self.area_priority.index(x.area.name))
        priority = [t for t in priority if t.area == priority[0].area]
        return priority if priority else list(self.player.territories)


    def initial_placement(self, empty, available):
        if empty:
            empty = sorted(empty, key=lambda x: self.area_priority.index(x.area.name))
            return empty[0]
        else:
            return random.choice(self.priority())

    def reinforce(self, available):
        priority = self.priority()
        result = collections.defaultdict(int)
        while available:
            result[random.choice(priority)] += 1
            available -= 1
        return result

    def attack(self):
        return([])


    def freemove(self):
        srcs = sorted([t for t in self.player.territories if not t.border],
                      key=lambda x: x.forces)
        if srcs:
            src = srcs[-1]
            n = src.forces - 1
            return (src, self.priority()[0], n)
        return None
