from random import choice
from kaggle_environments.envs.halite.helpers import *
import numpy as np

# Function stolen from notebook
def agent(obs, config):

    board = Board(obs, config)
    current_player = board.current_player

    ships = current_player.ships
    shipyards = current_player.shipyards


    # Spawn a ship      
    if len(ships) == 0:
        shipyards[0].next_action = ShipyardAction.SPAWN
    # Convert if only 1 shipyard
    elif len(shipyards) == 0:
        ships[0].next_action = ShipAction.CONVERT
    else:
        ships[0].next_action = minimax(board)
    # elif board.step == 1:
    #     ship_yard = current_player.shipyards[0]
    #     ship_yard.next_action = ShipyardAction.SPAWN
    # else:
    #     if len(current_player.ships) > 0:
    #         action = minimax(board, current_player)
    #         current_player.next_action = action
    #         print(action)
    #     elif len(current_player.shipyards) > 0:
    #         ship_yard = current_player.shipyards[0]
    #         ship_yard.next_action = ShipyardAction.SPAWN
    # except:
    #     print("error")
    #     print(current_player.ships)
    #     pass

    return current_player.next_actions

def argmax(li):
    m = float('-inf')
    ind = -1
    for i, item in enumerate(li):
        if item > m:
            m = item
            ind = i
    return ind

# Assume all are working against you
def minimax(board):
    current_player = board.current_player
    actions = [None, ShipAction.NORTH, ShipAction.EAST, ShipAction.SOUTH,ShipAction.WEST]

    for ship in current_player.ships:
        u = []
        for action in actions:
            ship.next_action = action
            b_prime = board.next()
            u.append(heuristic(b_prime))
    # print(u)
    return actions[argmax(u)]


def total_cargo(player):
    total = 0
    for ship in player.ships:
        total += ship.halite
    return total

def heuristic(board):
    h = 0
    deposited = board.current_player.halite
    h += deposited
    cargo = total_cargo(board.current_player)
    h += max(cargo, 500)

    
    # random correction
    h += np.random.rand()
    return 