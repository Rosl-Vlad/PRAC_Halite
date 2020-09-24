from kaggle_environments.envs.halite.helpers import *
import numpy as np

INF = 20
BOARD_SIZE = 21


class MyBoard:
    def __init__(self):
        self.board = np.ones((BOARD_SIZE, BOARD_SIZE))

    def reserved_cell(self, position):
        self.board[position] = INF

    def return_cell(self, position):
        self.board[position] = 1

    def return_all_cell(self):
        self.board = np.ones((BOARD_SIZE, BOARD_SIZE))

    def check_reserved_cell(self, position):
        if self.board[position] == INF:
            return True
        return False

    def make_a_way(self, destination, ship_location):
        if ship_location[1] - destination[1] > 0:
            return ShipAction.SOUTH, tuple(list(map(lambda x: x % BOARD_SIZE, ship_location + (0, -1))))

        if ship_location[1] - destination[1] < 0:
            return ShipAction.NORTH, tuple(list(map(lambda x: x % BOARD_SIZE, ship_location + (0, 1))))

        if ship_location[0] - destination[0] > 0:
            return ShipAction.WEST, tuple(list(map(lambda x: x % BOARD_SIZE, ship_location + (-1, 0))))

        if ship_location[0] - destination[0] < 0:
            return ShipAction.EAST, tuple(list(map(lambda x: x % BOARD_SIZE, ship_location + (1, 0))))

        return None, ship_location
