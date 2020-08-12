from kaggle_environments import make
from kaggle_environments.envs.halite.helpers import *
from tqdm import tqdm
from utils import Func
import numpy as np
import itertools
import random
import math

from model import ParseState, Policy


def discrete_halite(h, k=5):
    return int(math.ceil(1.0*h/env.configuration['maxCellHalite'] * k))

def discrete_player_halite(h, cost):
    return min(int(h/cost), 20)


class State:

    def __init__(self, s=None):


        if s is None:
            self.rep = None
            return

        step = s['step']
        
        # halite map
        halite = tuple(discrete_halite(h, k=3) for h in s['halite'])

        me = s['players'][0]
        my_halite, my_shipyards, my_ships = tuple(me)
        my_halite = discrete_player_halite(my_halite, env.configuration['convertCost'])

        my_shipyards_set = set()

        for shipyard_pos in my_shipyards.values():
            my_shipyards_set.add(shipyard_pos)
        
        my_ships_set = set()

        for ship_pos in my_ships.values():
            my_ships_set.add(ship_pos[0])


        opponents = s['players'][1:]
        opponent_ships = set()
        opponent_shipyards = set()

        for opponent in opponents:
            opp_halite, opp_shipyards, opp_ships = tuple(opponent)
            for shipyard_pos in opp_shipyards.values():
                opponent_shipyards.add(shipyard_pos)
            for ship_pos in opp_ships.values():
                opponent_ships.add((ship_pos[0], State.compass(halite, ship_pos[0])))

        self.rep = (step, frozenset(my_shipyards_set), frozenset(my_ships_set), frozenset(opponent_shipyards), frozenset(opponent_ships))
    
    @staticmethod
    def compass(halite, ship_pos):

        w = int(math.sqrt(len(halite)))

        # North, East, South, West
        points = State.cardinal_offsets(ship_pos, w)
 
        scores = []
        for i, p in enumerate(points):
            d_score = 0
            for q, h in enumerate(halite):
                d = State.toroid_l1_1d(p, q, w) + 1.0
                d_score += h/d**2
            scores.append((i, d_score))
        scores.sort(key=lambda x: x[1])

        compass = [0, 0, 0, 0]

        for i, (d, s) in enumerate(scores):
            compass[d] = i

        # h = halite[points[0]]

        return tuple(compass)

    @staticmethod
    def cardinal_offsets(pos, w):
        x = int(pos%w)
        y = int(pos/w)

        points = []

        x_ = x
        y_ = w - 1 if y == 0 else y - 1
        points.append(y_ * w + x_)

        x_ = 0 if x == w-1 else x+1
        y_ = y
        points.append(y_ * w + x_)

        x_ = x
        y_ = 0 if y == w-1 else y + 1
        points.append(y_ * w + x_)

        x_ = w - 1 if x == 0 else x - 1
        y_ = y
        points.append(y_ * w + x_)
        
        return points
    
    @staticmethod
    def toroid_l1_1d(p, q, w):
        x_p =  int(p%w)
        y_p =  int(p/w)
        x_q = int(q%w)
        y_q = int(q/w)
        dx = abs(x_p - x_q)
        dx = dx if dx < w/2.0 else w - dx
        dy = abs(y_p - y_q)
        dy = dy if dy < w/2.0 else w - dy
        return dy + dx

    def __hash__(self):
        return self.rep.__hash__()
    
    def __str__(self):
        return self.rep.__str__()

class Action:
    SHIP_ACTIONS = (None, "NORTH", "EAST", "SOUTH", "WEST", "CONVERT")
    SHIPYARD_ACTIONS = (None, "SPAWN")
    MAX_SHIPS = 2
    MAX_SHIPYARDS = 1

    SHIP_TYPE = 0
    SHIPYARD_TYPE = 1

    
    @classmethod
    def act_map_to_act(cls, s, a):
        me = s['players'][0]
        rep = set()
        for uid, act in a.items():
            if uid in me[1]:
                rep.add((me[1][uid], cls.SHIPYARD_TYPE, act))
            elif uid in me[2]:
                rep.add((me[2][uid][0], cls.SHIP_TYPE, act))
        return frozenset(rep)
    
    @classmethod
    def legal_actions(cls, state):
        me = s['players'][0]

        num_ships = len(me[2])
        num_shipyards = len(me[1])

        args = [cls.SHIP_ACTIONS] * num_ships + [cls.SHIPYARD_ACTIONS] * num_shipyards

        typ = [cls.SHIP_TYPE] * num_ships + [cls.SHIPYARD_TYPE] * num_shipyards

        pos = [pos for pos, _ in me[2].values()] + [pos for pos in me[1].values()]
        uids = [uid for uid in me[2].keys()] + [uid for uid in me[1].keys()]
        valid = []
        uid_map = []
        for act in itertools.product(*args):
            a = frozenset(zip(pos, typ, act))
            if cls.validate_action(state, a):                
                valid.append(a)
                uid_map.append(dict(zip(uids, act)))
        return valid, uid_map


    @classmethod
    def validate_action(cls, state, action):
        me = s['players'][0]
        ctx_typ = set()
        spawn = 0
        convert = 0
        for ctx, typ, act in action:
            if act == "SPAWN":
                spawn += 1
            elif act == "CONVERT":
                convert+=1
            ctx_typ.add((ctx, typ))

        if len(ctx_typ) != len(action) or spawn+convert > int(me[0] / 500):
            return False
        if spawn + len(me[2]) > cls.MAX_SHIPS or convert + len(me[1]) > cls.MAX_SHIPYARDS:
            return False
        return True


class MultiAgent:

    def __init__(self, epsilon = .2, alpha = .4, gamma = 1):
 
        self.Q = Func()
        self.N = Func()
        
        self.epsilon = epsilon
        self.alpha = lambda n: alpha * (n+1)
        self.gamma = gamma
    
    def get_actions(self, state, eval=False):
        s = State(state)

        actions, actions_dict = Action.legal_actions(s)
        if eval is not False or np.random.rand() > self.epsilon:
            return actions_dict[self.Q.argmax((s, ), actions)]
        else:
            return random.choice(actions_dict)
    
    def learn(self, s, a, r, s_):
        os = s
        os_ = s_

        s = State(s)
        s_ = State(s_)    

        a_, _ = Action.legal_actions(os_)
        a = Action.act_map_to_act(os, a)
        q_max = self.Q.max((s_, ), a_) if s_ is not None else 0
        self.Q[(s, a)] += self.alpha(self.N[(s,a)]) * (r + self.gamma * q_max - self.Q[(s, a)])
        self.N[(s,a)] += 1





if __name__ == "__main__":

    env = make("halite", debug=True)
    print(env.configuration)

    Q_ship = Func()
    Q_shipyard = Func()

    N = Func()

    # Training agent in first position (player 1) against the default random agent.
    trainer = env.train([None, "random", "random", "random"])

    s = trainer.reset()

    policy = Policy(env.configuration)
    
    T = 4000
    # alpha = .1
    # gamma = 1
    # epsilon = .2

    # pbar = tqdm(total=T)
    # print(State(s))


    # print(Action.legal_actions(s))
    # ship = Ship()

    # shipyard = Shipyard()

    # print(ship.state(s))
    # T = 4000
    # alpha = .1
    # gamma = 1
    # epsilon = .2

    # agent = MultiAgent(alpha=alpha, gamma=gamma, epsilon=epsilon)
    # pbar = tqdm(total=T)

    # history = []

    # for i in range(T):
    #     s = trainer.reset()
    #     # print(State.compass(s['halite'], 0))
    #     # print(env.render(mode='ansi'))
    #     # break
    #     r = None
    #     terminal = False

    #     r_sum = 5000

    #     while not terminal:
    #         a = agent.get_actions(s)
           
    #         s_, r, terminal, info = trainer.step({k: v for k, v in a.items() if v is not None})

    #         if terminal:
    #             s_ = None
        
    #         agent.learn(s, a, r, s_)
    #         r_sum += r
    #         s = s_
    #     history.append(r_sum)
    #     if len(history) > 1000:
    #         history = history[1:]
    #     pbar.write('Episode {}: {}, AVG: {}'.format(i, r_sum, 1.0 * sum(history) / len(history)))

    #     if i % 100 == 0:
    #         f = open('./index.html', 'w')
    #         f.write(env.render(mode='html', width=400, height=600))
    #         f.close()

    #     pbar.update()

            
            
