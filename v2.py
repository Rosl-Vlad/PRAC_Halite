from kaggle_environments.envs.halite.helpers import *
import numpy as np

INF = 20
BOARD_SIZE = 21
FINISH_STEP = 385
SHIP_COUNT = 10

class CollectorShip(object):
    def __init__(self, ship):
        self.id = ship.id
        self.cell = ship.cell
        self.ship = ship
        self.halite = ship.halite
        self.position = ship.position
        self.nearby_halite_deposits = dict()
        self.next_action = None
        self.action = False
        self.priority = []

    def __repr__(self):
        return "collector"

    def collect(self):
        self.ship.next_action = None

    def get_back(self):
        self.nearby_halite_deposits[self.position] = "BACK"

    def check_fullness(self, critical_value):
        if self.halite >= critical_value:
            return True
        return False

    def commit_cell_info(self, cell, deep):
        if self.nearby_halite_deposits.get(cell.position) is None:
            self.nearby_halite_deposits[cell.position] = {"halite": cell.halite, "steps": deep, "prior": 100}

    def set_action(self, action):
        self.next_action = action
        self.ship.next_action = action

    def search_halite(self, board, max_deep):
        for i in range(max_deep):
            for j in range(max_deep):
                self.commit_cell_info(board[(self.position[0] + i, self.position[1] + j)], i + j)
                self.commit_cell_info(board[(self.position[0] - i, self.position[1] + j)], i + j)
                self.commit_cell_info(board[(self.position[0] + i, self.position[1] - j)], i + j)
                self.commit_cell_info(board[(self.position[0] - i, self.position[1] - j)], i + j)

        self.prior_cell(0.8)

    def prior_cell(self, greed_val):
        def f(x):
            return x[1]["halite"] / (x[1]["steps"] * greed_val + 1)

        self.nearby_halite_deposits = {k: v for k, v in sorted(self.nearby_halite_deposits.items(), key=f, reverse=True)}
        for i, position in enumerate(self.nearby_halite_deposits.keys()):
            self.priority.append(position)


class HunterShip:
    def __init__(self, ship, enemies):
        self.id = ship.id
        self.cell = ship.cell
        self.ship = ship
        self.halite = ship.halite
        self.position = ship.position
        self.nearby_enemy_ship = dict()
        self.next_action = None
        self.action = False
        self.priority = []
        self.enemies = enemies

    def search_nearest_enemies_ship(self):
        def calculate_steps(a, b):
            return abs(sum(np.array(a) - np.array(b)))

        for enemy in self.enemies:
            for enemy_ship in enemy.ships:
                self.nearby_enemy_ship[enemy_ship.position] = {"halite": enemy_ship.halite,
                                                               "steps": calculate_steps(self.position,
                                                                                        enemy_ship.position)}

        self.prior_target(1)

    def prior_target(self, greed_val):
        def f(x):
            return x[1]["halite"] / (x[1]["steps"] * greed_val + 1)

        self.nearby_enemy_ship = {k: v for k, v in sorted(self.nearby_enemy_ship.items(), key=f, reverse=True)}
        for i, position in enumerate(self.nearby_enemy_ship.keys()):
            self.priority.append(position)

    def check_fullness(self, critical_value):
        if self.halite >= critical_value:
            return True
        return False

    def set_action(self, action):
        self.next_action = action
        self.ship.next_action = action


class Shipyard:
    def __init__(self, shipyard):
        self.shipyard = shipyard
        self.position = shipyard.position
        self.next_action = None

    def set_action(self):
        self.shipyard.next_action = self.next_action


class Commander:
    def __init__(self, player, board):
        self.ships = []
        self.shipyard = None
        self.my_board = MyBoard()
        self.board = board
        self.current_player = player
        self.target_cell = set()

    def add_ship(self, ship):
        self.ships.append(ship)

    def add_shipyard(self, shipyard):
        self.shipyard = shipyard

    def clear_state(self):
        self.my_board.return_all_cell()
        self.ships = []
        self.shipyard = None
        self.target_cell = set()

    def reserved_enemy_cells(self):
        for player in self.board.opponents:
            for shipyard in player.shipyards:
                self.my_board.reserved_cell(shipyard.position)
            for ship in player.ships:
                if ship.halite < 500:
                    self.my_board.reserved_cell(ship.position)
                    self.my_board.reserved_cell(tuple(list(map(lambda x: x % BOARD_SIZE, ship.position + (1, 0)))))
                    self.my_board.reserved_cell(tuple(list(map(lambda x: x % BOARD_SIZE, ship.position + (-1, 0)))))
                    self.my_board.reserved_cell(tuple(list(map(lambda x: x % BOARD_SIZE, ship.position + (0, -1)))))
                    self.my_board.reserved_cell(tuple(list(map(lambda x: x % BOARD_SIZE, ship.position + (0, 1)))))

    def shipyard_action(self):
        if self.shipyard is not None:
            if self.my_board.check_reserved_cell(self.shipyard.position) is not True\
                    and self.current_player.halite > 600 \
                    and self.board.step <= FINISH_STEP \
                    and len(self.current_player.ships) < SHIP_COUNT\
                    and self.board.step < 300:

                self.my_board.reserved_cell(self.shipyard.position)
                self.shipyard.next_action = ShipyardAction.SPAWN
                self.shipyard.set_action()

    def ship_action(self):
        if self.board.step == 398:
            for ship in self.ships:
                ship.set_action(ShipAction.CONVERT)
                ship.action = True
            return

        for ship in self.ships:
            if self.shipyard is None:
                ship.set_action(ShipAction.CONVERT)
                self.shipyard = False
                ship.action = True
                continue

            if self.board.step > FINISH_STEP:
                select_position, action = self.my_board.get_dir(ship.position, self.shipyard.position)
                ship.set_action(action)
                ship.action = True
                continue

            elif ship.check_fullness(350) is True:
                if self.shipyard is False:
                    break
                select_position, action = self.my_board.get_dir(ship.position, self.shipyard.position)
                if self.my_board.check_reserved_cell(select_position) is not True:
                    #self.my_board.reserved_cell(self.shipyard.position)
                    self.my_board.reserved_cell(select_position)
                    ship.set_action(action)
                    ship.action = True
                    continue

        for ship in self.ships:
            if ship.action is True:
                continue

            if repr(ship) == "collector":
                ship.search_halite(self.board, 9)
            else:
                ship.search_nearest_enemies_ship()

            for position in ship.priority:
                select_position, action = self.my_board.get_dir(ship.position, position)
                if self.my_board.check_reserved_cell(select_position) is not True and position not in self.target_cell:
                    self.target_cell.add(position)
                    self.my_board.reserved_cell(select_position)
                    ship.set_action(action)
                    ship.action = True
                    break
            ship.action = True


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

    def get_dir(self, a, b):
        if a == b:
            return a, None

        from_x, from_y = divmod(a[0], BOARD_SIZE - 1), divmod(a[1], BOARD_SIZE - 1)
        to_x, to_y = divmod(b[0], BOARD_SIZE - 1), divmod(b[1], BOARD_SIZE - 1)
        a_north = a[0], divmod(a[1] + 1, BOARD_SIZE - 1)[1]
        a_south = a[0], divmod(a[1] - 1, BOARD_SIZE - 1)[1]
        a_east = divmod(a[0] + 1, BOARD_SIZE - 1)[1], a[1]
        a_west = divmod(a[0] - 1, BOARD_SIZE - 1)[1], a[1]

        if from_y < to_y:
            if self.check_reserved_cell(a_north) is True:
                if from_x < to_x and self.check_reserved_cell(a_east) is True:
                    return a_east, ShipAction.EAST
                if from_x > to_x and self.check_reserved_cell(a_west) is True:
                    return a_west, ShipAction.WEST
            return a_north, ShipAction.NORTH
        if from_y > to_y:
            if self.check_reserved_cell(a_south) is True:
                if from_x < to_x and self.check_reserved_cell(a_east) is True:
                    return a_east, ShipAction.EAST
                if from_x > to_x and self.check_reserved_cell(a_west) is True:
                    return a_west, ShipAction.WEST
            return a_south, ShipAction.SOUTH
        if from_x < to_x:
            if self.check_reserved_cell(a_east) is True:
                if from_y < to_y and self.check_reserved_cell(a_north) is True:
                    return a_north, ShipAction.NORTH
                if from_y > to_y and self.check_reserved_cell(a_south) is True:
                    return a_south, ShipAction.SOUTH
            return a_east, ShipAction.EAST
        if from_x > to_x:
            if self.check_reserved_cell(a_west) is True:
                if from_y < to_y and self.check_reserved_cell(a_north) is True:
                    return a_north, ShipAction.NORTH
                if from_y > to_y and self.check_reserved_cell(a_south) is True:
                    return a_south, ShipAction.SOUTH
            return a_west, ShipAction.WEST
        return a, None


def agent(obs, config):
    board = Board(obs, config)
    me = board.current_player

    commander = Commander(me, board)
    commander.clear_state()
    commander.reserved_enemy_cells()
    for ship in me.ships:
        if board.step < 399:
            commander.add_ship(CollectorShip(ship))
        else:
            commander.add_ship(HunterShip(ship, board.opponents))

    for shipyard in me.shipyards:
        commander.add_shipyard(Shipyard(shipyard))

    commander.shipyard_action()
    commander.ship_action()

    return me.next_actions
