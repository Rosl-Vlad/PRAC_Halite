from kaggle_environments.envs.halite.helpers import *
import numpy as np
from munkres import Munkres, print_matrix

BOARD_SIZE = 21
enemies_ships_pos_halite = None
enemies_shipyards_pos = None
halite_position = None
matrix_of_reward = None
current_count_of_ships = None


def enemies_ships(board):
    global enemies_ships_pos_halite
    enemies_ships_pos_halite = np.zeros((BOARD_SIZE, BOARD_SIZE))
    for player in board.opponents:
        for ship in player.ships:
            if ship.halite == 0:
                enemies_ships_pos_halite[ship.position] = 0
            else:
                enemies_ships_pos_halite[ship.position] = ship.halite


def enemies_sy(board):
    global enemies_shipyards_pos
    enemies_shipyards_pos = np.zeros((BOARD_SIZE, BOARD_SIZE))
    for player in board.opponents:
        for shipyard in player.shipyards:
            enemies_shipyards_pos[shipyard.position] = 1


def total_halite(board):
    global halite_position
    halite_position = np.zeros((BOARD_SIZE, BOARD_SIZE))
    for pos, cell_info in board.cells.items():
        halite_position[pos] = cell_info.halite


def sub_array(position):
    return np.array(tuple(list(map(lambda x: x % BOARD_SIZE, position))))


def sub_tuple(position):
    return tuple(list(map(lambda x: x % BOARD_SIZE, position)))


def calculate_value_per_ship(halite_on_ship, position, depth):
    possibility = {}
    for i in range(depth):
        for j in range(depth):
            pos1 = sub_tuple(position + (i, j))
            pos2 = sub_tuple(position + (-i, j))
            pos3 = sub_tuple(position + (i, -j))
            pos4 = sub_tuple(position + (-i, -j))

            def calculate(pos, steps):
                def nearest_enemy_ships(pos_, depth_):
                    pos_ = np.array(pos_)
                    dangerous_ = {}
                    for i_ in range(depth_ + 1):
                        for j_ in range(depth_ + 1):
                            halite_ship_1 = enemies_ships_pos_halite[sub_tuple(pos_ + (i, j))] < halite_on_ship
                            halite_ship_2 = enemies_ships_pos_halite[sub_tuple(pos_ + (i, -j))] < halite_on_ship
                            halite_ship_3 = enemies_ships_pos_halite[sub_tuple(pos_ + (-i, j))] < halite_on_ship
                            halite_ship_4 = enemies_ships_pos_halite[sub_tuple(pos_ + (-i, -j))] < halite_on_ship
                            if halite_ship_1 or halite_ship_2 or halite_ship_3 or halite_ship_4:
                                dangerous_[abs(i_ + j_)] = True
                            else:
                                dangerous_[abs(i_ + j_)] = False
                    return dangerous_

                dangerous = nearest_enemy_ships(pos, 1)
                halite_cell = halite_position[pos]
                enemy_ship = {"halite": enemies_ships_pos_halite[pos], "dangerous": enemies_ships_pos_halite[pos] < halite_on_ship}
                enemy_sy = enemies_shipyards_pos[pos]
                return {"enemy_shipyard": enemy_sy,
                        "enemy_ship": enemy_ship,
                        "halite": halite_cell,
                        "nearest_enemy": dangerous,
                        "steps": steps}

            possibility[pos1] = calculate(pos1, abs(position[0] - pos1[0]) + abs(position[1] - pos1[1]))
            possibility[pos2] = calculate(pos2, abs(position[0] - pos2[0]) + abs(position[1] - pos2[1]))
            possibility[pos3] = calculate(pos3, abs(position[0] - pos3[0]) + abs(position[1] - pos3[1]))
            possibility[pos4] = calculate(pos4, abs(position[0] - pos4[0]) + abs(position[1] - pos4[1]))
    #print(possibility)
    return possibility


def func_investment(possibility, idx_ship):
    for pos, info in possibility.items():
        if (info["nearest_enemy"][1] and info["steps"] < 3) or info["enemy_ship"]["dangerous"]:
            matrix_of_reward[pos][idx_ship] = -1
        elif info["enemy_ship"]["dangerous"] is False:
            matrix_of_reward[pos][idx_ship] = (info["halite"] + info["enemy_ship"]["halite"]) / (info["step"] + 1)
        else:
            #print(pos, info["halite"] / (info["steps"] + 1))
            matrix_of_reward[pos][idx_ship] = info["halite"] / (info["steps"] + 1)


def convert_matrix_of_reward_in_two_dimension():
    c = 1
    new_board = np.zeros((BOARD_SIZE * BOARD_SIZE, BOARD_SIZE * BOARD_SIZE))
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            for idx_ship, reward in enumerate(matrix_of_reward[(i, j)]):
                new_board[c * i + j][idx_ship] = reward
        c += 1
    return new_board


def aggregate_possibility(ships_possibility):
    global matrix_of_reward
    matrix_of_reward = np.ones((BOARD_SIZE, BOARD_SIZE, BOARD_SIZE * BOARD_SIZE))
    for idx, possibility in enumerate(ships_possibility.values()):
        func_investment(possibility, idx)

    return convert_matrix_of_reward_in_two_dimension()


def search_value(board):
    my_ships = board.current_player.ships
    ships_possibility = {}
    for ship in my_ships:
        ships_possibility[ship] = calculate_value_per_ship(ship.halite, ship.position, BOARD_SIZE)
    mat = aggregate_possibility(ships_possibility)
    m = Munkres()
    indexes = m.compute(mat)


def agent(obs, config):
    global current_count_of_ships
    board = Board(obs, config)
    me = board.current_player
    current_count_of_ships = len(me.ships)

    enemies_sy(board)
    enemies_ships(board)
    total_halite(board)
    search_value(board)
    return me.next_actions
