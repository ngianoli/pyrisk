from ai import AI
import random
import collections

class CleverAI(AI):
    """
    CleverAI: improve BetterAI by forcing attacks when troops are above 100.
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
        for t in self.player.territories:
            if t.forces > 1:
                #adjacent = [a for a in t.connect if a.owner != t.owner and t.forces >= a.forces + 3]
                adjacent = [a for a in t.connect if a.owner != t.owner]
                #remove last condition to force attack

                easy_conquest = [a for a in t.connect if (a.owner != t.owner) and ( (t.forces > 3*a.forces) or ((t.forces > 30) and (5 > a.forces)) )]
                for adj in easy_conquest:
                    yield (t.name, adj, None, lambda x: 3 )

                if (t.forces > 100) & (len(adjacent) > 0):
                    yield (t.name, adjacent[0].name, None, None)

                if len(adjacent) == 1:
                        yield (t.name, adjacent[0].name,
                               lambda a, d: a > d, None)
                else:
                    total = sum(a.forces for a in adjacent)
                    for adj in adjacent:
                        yield (t, adj, lambda a, d: a > d + total - adj.forces + 3,
                               lambda a: 1)

    def freemove(self):
        srcs = sorted([t for t in self.player.territories if not t.border],
                      key=lambda x: x.forces)
        if srcs:
            src = srcs[-1]
            n = src.forces - 1
            return (src, self.priority()[0], n)
        return None
