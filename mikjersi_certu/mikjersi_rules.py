#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""mikjersi_rules.py implements the rules engine for the MIKJERSI boardgame."""

__version__ = "0.0.0"

_COPYRIGHT_AND_LICENSE = """
MIKJERSI-CERTU implements a GUI and a rules engine for the MIKJERSI boardgame.

Copyright (C) 2021 Lucas Borboleta (lucas.borboleta@free.fr).

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see <http://www.gnu.org/licenses>.
"""


import array
import collections
import copy
import enum
import itertools
import math
import os
import random
import re
import time

import mcts

_do_debug = False


OMEGA = 1_000.
OMEGA_2 = OMEGA**2


def chunks(sequence, chunk_count):
    """ Yield chunck_count successive chunks from sequence"""
    chunk_size = int(len(sequence) / chunk_count)
    chunk_end = 0
    for _ in range(0, chunk_count - 1):
        chunk_start = chunk_end
        chunk_end = chunk_start + chunk_size
        yield sequence[chunk_start : chunk_end]
    yield sequence[chunk_end:]


def partition(predicate, iterable):
    """Use a predicate to partition entries into false entries and true entries"""
    (t1, t2) = itertools.tee(iterable)
    return (itertools.filterfalse(predicate, t1), filter(predicate, t2))


def cell_distance(position_uv_1, position_uv_2):
    (u1, v1) = position_uv_1
    (u2, v2) = position_uv_2
    distance = math.fabs(u1 - u2) + math.fabs(v1 - v2)
    return distance


class JersiMcts(mcts.mcts):

    def __init__(self,*args, **kwargs):
        super().__init__(*args,** kwargs)


    def getBestActions(self):
        bestActions = []
        bestValue = -math.inf
        node = self.root
        currentPlayer = node.state.getCurrentPlayer()
        for (action, child) in node.children.items():
            childValue = currentPlayer*child.totalReward/child.numVisits
            if childValue > bestValue:
                bestValue = childValue
                bestActions = [action]
            elif childValue == bestValue:
                bestActions.append(action)
        return bestActions


@enum.unique
class Capture(enum.Enum):
    KING_CUBE = enum.auto()
    KING_STACK = enum.auto()
    NONE = enum.auto()
    SOME_CUBE = enum.auto()
    SOME_STACK = enum.auto()


@enum.unique
class CubeSort(enum.Enum):
    FOOL = enum.auto()
    KING = enum.auto()
    MOUNTAIN = enum.auto()
    PAPER = enum.auto()
    ROCK = enum.auto()
    SCISSORS = enum.auto()
    WISE = enum.auto()


@enum.unique
class CubeStatus(enum.IntEnum):
    ACTIVATED = -121
    CAPTURED = -122
    RESERVED = -123
    UNUSED = -124


@enum.unique
class CellDirection(enum.IntEnum):
    PHI_000 = 0
    PHI_090 = 1
    PHI_180 = 2
    PHI_270 = 3


@enum.unique
class Null(enum.IntEnum):
    CUBE = -101
    CELL = -102


class Player(enum.IntEnum):
    WHITE = 0
    BLACK = 1

    @staticmethod
    def name(player):
        if player == Player.WHITE:
            return "white"

        elif player == Player.BLACK:
            return "black"

        else:
            assert False


@enum.unique
class Reward(enum.IntEnum):
    WIN = 1
    DRAW = 0
    LOSS = -1

    assert LOSS < DRAW < WIN
    assert DRAW == 0
    assert LOSS + WIN == DRAW


@enum.unique
class TerminalCase(enum.Enum):

    BLACK_ARRIVED = enum.auto()
    BLACK_CAPTURED = enum.auto()
    BLACK_BLOCKED = enum.auto()

    WHITE_ARRIVED = enum.auto()
    WHITE_CAPTURED = enum.auto()
    WHITE_BLOCKED = enum.auto()

    ZERO_CREDIT = enum.auto()


class Cube:

    __slots__ = ('name', 'label', 'sort', 'player', 'index')

    __all_sorted_cubes = []
    __init_done = False
    __king_index = None
    __name_to_cube = {}
    __opposite_index = None
    __sort_and_player_to_label = {}

    all = None # shortcut to Cube.get_all()
    black_king_index = None
    white_king_index = None


    def __init__(self, name, label, sort, player):
        """Create a cube and check its properties"""

        assert name not in Cube.__name_to_cube
        assert len(name) == 2
        assert len(label) == 1
        assert label == name[0]

        assert sort in CubeSort
        assert player in Player

        if player == Player.WHITE:
            assert label == label.upper()
        elif player == Player.BLACK:
            assert label == label.lower()
        else:
            assert False

        if (sort, player) not in Cube.__sort_and_player_to_label:
            Cube.__sort_and_player_to_label[(sort, player)] = label
        else:
            assert Cube.__sort_and_player_to_label[(sort, player)] == label

        self.name = name
        self.label = label
        self.sort = sort
        self.player = player
        self.index = None

        Cube.__name_to_cube[self.name] = self


    def __str__(self):
        return f"Cube({self.name}, {self.label}, {self.sort}, {self.player}, {self.index})"


    def beats(self, other):

        if self.player == other.player:
            does_beat = False

        else:

            if self.sort in (CubeSort.KING, CubeSort.WISE, CubeSort.MOUNTAIN):
                does_beat = False

            elif other.sort == CubeSort.MOUNTAIN:
                does_beat = False

            elif self.sort == CubeSort.ROCK:
                does_beat = other.sort in (CubeSort.SCISSORS, CubeSort.FOOL, CubeSort.KING, CubeSort.WISE)

            elif self.sort == CubeSort.PAPER:
                does_beat = other.sort in (CubeSort.ROCK, CubeSort.FOOL, CubeSort.KING, CubeSort.WISE)

            elif self.sort == CubeSort.SCISSORS:
                does_beat = other.sort in (CubeSort.PAPER, CubeSort.FOOL, CubeSort.KING, CubeSort.WISE)

            elif self.sort == CubeSort.FOOL:
                does_beat = other.sort in (CubeSort.ROCK, CubeSort.PAPER, CubeSort.SCISSORS, CubeSort.FOOL, CubeSort.KING)

            else:
                assert False

        return does_beat


    @staticmethod
    def get(name):
        return Cube.__name_to_cube[name]


    @staticmethod
    def get_all():
        return Cube.__all_sorted_cubes


    @staticmethod
    def get_king_index(player):
        return Cube.__king_index[player]


    @staticmethod
    def get_opposite_index(cube_index):
        return Cube.__opposite_index[cube_index]


    @staticmethod
    def init():
        if not Cube.__init_done:
            Cube.__create_cubes()
            Cube.__create_all_sorted_cubes()
            Cube.__create_king_index()
            Cube.__create__opposite_index()
            Cube.__init_done = True


    @staticmethod
    def show_all():
        for cube in Cube.__all_sorted_cubes:
            print(cube)


    @staticmethod
    def __create_all_sorted_cubes():
        for name in sorted(Cube.__name_to_cube.keys()):
            Cube.__all_sorted_cubes.append(Cube.__name_to_cube[name])

        for (index, cube) in enumerate(Cube.__all_sorted_cubes):
            cube.index = index

        Cube.all = Cube.__all_sorted_cubes


    @staticmethod
    def __create_king_index():
        Cube.__king_index = array.array('b', [Null.CUBE for _ in Player])
        Cube.__king_index[Player.WHITE] = Cube.get('K1').index
        Cube.__king_index[Player.BLACK] = Cube.get('k1').index

        Cube.white_king_index = Cube.get_king_index(Player.WHITE)
        Cube.black_king_index = Cube.get_king_index(Player.BLACK)


    @staticmethod
    def __create__opposite_index():
        Cube.__opposite_index = array.array('b', [Null.CUBE for _ in Cube.all])
        for cube in Cube.all:
            for opposite in Cube.all:
                if opposite.sort == cube.sort and opposite.player != cube.player:
                    Cube.__opposite_index[cube.index] = opposite.index
                    break


    @staticmethod
    def __create_cubes():

        Cube(name='K1', label='K', sort=CubeSort.KING, player=Player.WHITE)

        Cube(name='F1', label='F', sort=CubeSort.FOOL, player=Player.WHITE)

        Cube(name='W1', label='W', sort=CubeSort.WISE, player=Player.WHITE)

        Cube(name='R1', label='R', sort=CubeSort.ROCK, player=Player.WHITE)

        Cube(name='P1', label='P', sort=CubeSort.PAPER, player=Player.WHITE)

        Cube(name='S1', label='S', sort=CubeSort.SCISSORS, player=Player.WHITE)

        Cube(name='M1', label='M', sort=CubeSort.MOUNTAIN, player=Player.WHITE)

        Cube(name='k1', label='k', sort=CubeSort.KING, player=Player.BLACK)

        Cube(name='f1', label='f', sort=CubeSort.FOOL, player=Player.BLACK)

        Cube(name='w1', label='w', sort=CubeSort.WISE, player=Player.BLACK)

        Cube(name='r1', label='r', sort=CubeSort.ROCK, player=Player.BLACK)

        Cube(name='p1', label='p', sort=CubeSort.PAPER, player=Player.BLACK)

        Cube(name='s1', label='s', sort=CubeSort.SCISSORS, player=Player.BLACK)

        Cube(name='m1', label='m', sort=CubeSort.MOUNTAIN, player=Player.BLACK)


class Cell:

    __slots__ = ('name', 'position_uv', 'reserve', 'player', 'index')

    __all_active_indices = []
    __all_indices = []
    __all_sorted_cells = []
    __init_done = False
    __king_begin_indices = []
    __king_end_indices = []
    __layout = []
    __name_to_cell = {}
    __next_fst_indices = []
    __next_snd_indices = []
    __position_uv_to_cell = {}

    all = None # shortcut to Cell.get_all()


    def __init__(self, name, position_uv, reserve=False, player=None):

        assert name not in Cell.__name_to_cell
        assert len(position_uv) == 2
        assert reserve in [True, False]
        assert position_uv not in Cell.__position_uv_to_cell
        assert player is None or player in Player

        self.name = name
        self.position_uv = position_uv
        self.reserve = reserve
        self.player = player
        self.index = None

        Cell.__name_to_cell[self.name] = self
        Cell.__position_uv_to_cell[position_uv] = self


    def __str__(self):
        return f"Cell({self.name}, {self.position_uv}, {self.reserve}), {self.index}"


    @staticmethod
    def get(name):
        return Cell.__name_to_cell[name]


    @staticmethod
    def get_all():
        return Cell.__all_sorted_cells


    @staticmethod
    def get_all_active_indices():
        return Cell.__all_active_indices


    @staticmethod
    def get_all_indices():
        return Cell.__all_indices


    @staticmethod
    def get_king_begin_indices(player):
        return Cell.__king_begin_indices[player]


    @staticmethod
    def get_king_end_indices(player):
        return Cell.__king_end_indices[player]


    @staticmethod
    def get_layout():
        return Cell.__layout


    @staticmethod
    def get_next_fst_active_indices(cell_index):
        return [x for x in Cell.__next_fst_indices[cell_index] if x != Null.CELL]


    @staticmethod
    def get_next_fst_indices(cell_index, cell_dir):
        return Cell.__next_fst_indices[cell_index][cell_dir]


    @staticmethod
    def get_next_snd_indices(cell_index, cell_dir):
        return Cell.__next_snd_indices[cell_index][cell_dir]


    @staticmethod
    def init():
        if not  Cell.__init_done:
            Cell.__create_cells()
            Cell.__create_all_sorted_cells()
            Cell.__create_layout()
            Cell.__create_kings_cells()
            Cell.__create_delta_u_and_v()
            Cell.__create_next_cells()
            Cell.__init_done = True


    @staticmethod
    def show_all():
        for cell in Cell.__all_sorted_cells:
            print(cell)


    @staticmethod
    def __create_all_sorted_cells():
        for name in sorted(Cell.__name_to_cell.keys()):
            Cell.__all_sorted_cells.append(Cell.__name_to_cell[name])

        for (index, cell) in enumerate(Cell.__all_sorted_cells):
            cell.index = index

        for cell in Cell.__all_sorted_cells:
            Cell.__all_indices.append(cell.index)
            if not cell.reserve:
                Cell.__all_active_indices.append(cell.index)

        Cell.all = Cell.__all_sorted_cells


    @staticmethod
    def __create_delta_u_and_v():
        Cell.__delta_u = array.array('b', [+0, +1, +0, -1])
        Cell.__delta_v = array.array('b', [+1, +0, -1, +0])


    @staticmethod
    def __create_kings_cells():

        white_first_cells = ["a1", "a2", "a3", "a4", "a5"]
        black_first_cells = ["e1", "e2", "e3", "e4", "e5"]

        white_first_indices = array.array('b', map(lambda x: Cell.get(x).index, white_first_cells))
        black_first_indices = array.array('b', map(lambda x: Cell.get(x).index, black_first_cells))

        Cell.__king_begin_indices = [None for _ in Player]
        Cell.__king_end_indices = [None for _ in Player]

        Cell.__king_begin_indices[Player.WHITE] = white_first_indices
        Cell.__king_begin_indices[Player.BLACK] = black_first_indices

        Cell.__king_end_indices[Player.WHITE] = black_first_indices
        Cell.__king_end_indices[Player.BLACK] = white_first_indices


    @staticmethod
    def __create_layout():

        Cell.__layout = []

        Cell.__layout.append( (0, ["e1", "e2", "e3", "e4", "e5"]))
        Cell.__layout.append( (0, ["d1", "d2", "d3", "d4", "d5"]))
        Cell.__layout.append( (0, ["c1", "c2", "c3", "c4", "c5"]))
        Cell.__layout.append( (0, ["b1", "b2", "b3", "b4", "b5"]))
        Cell.__layout.append( (0, ["a1", "a2", "a3", "a4", "a5"]))


    @staticmethod
    def __create_next_cells():

        Cell.__next_fst_indices = [None for _ in Cell.__all_sorted_cells]
        Cell.__next_snd_indices = [None for _ in Cell.__all_sorted_cells]

        for (cell_index, cell) in enumerate(Cell.__all_sorted_cells):
            (cell_u, cell_v) = cell.position_uv

            Cell.__next_fst_indices[cell_index] = array.array('b', [Null.CELL for _ in CellDirection])
            Cell.__next_snd_indices[cell_index] = array.array('b', [Null.CELL for _ in CellDirection])

            if not cell.reserve:
                for cell_dir in CellDirection:
                    cell_delta_u = Cell.__delta_u[cell_dir]
                    cell_delta_v = Cell.__delta_v[cell_dir]

                    cell_fst_u = cell_u + 1*cell_delta_u
                    cell_fst_v = cell_v + 1*cell_delta_v

                    cell_snd_u = cell_u + 2*cell_delta_u
                    cell_snd_v = cell_v + 2*cell_delta_v

                    if (cell_fst_u, cell_fst_v) in Cell.__position_uv_to_cell:
                        cell_fst = Cell.__position_uv_to_cell[(cell_fst_u, cell_fst_v)]
                        if not cell_fst.reserve:
                            Cell.__next_fst_indices[cell_index][cell_dir] = cell_fst.index

                        if (cell_snd_u, cell_snd_v) in Cell.__position_uv_to_cell:
                            cell_snd = Cell.__position_uv_to_cell[(cell_snd_u, cell_snd_v)]
                            if not cell_snd.reserve:
                                Cell.__next_snd_indices[cell_index][cell_dir] = cell_snd.index


    @staticmethod
    def __create_cells():

        # Row "a"
        Cell('a1', (-2, -2))
        Cell('a2', (-1, -2))
        Cell('a3', (0, -2))
        Cell('a4', (1, -2))
        Cell('a5', (2, -2))

        Cell('A0', (-3, -3), reserve=True, player=Player.WHITE)
        Cell('A1', (-2, -3), reserve=True, player=Player.WHITE)
        Cell('A2', (-1, -3), reserve=True, player=Player.WHITE)
        Cell('A3', (0, -3), reserve=True, player=Player.WHITE)
        Cell('A4', (1, -3), reserve=True, player=Player.WHITE)
        Cell('A5', (2, -3), reserve=True, player=Player.WHITE)
        Cell('A6', (3, -3), reserve=True, player=Player.WHITE)

        # Row "b"
        Cell('b1', (-2, -1))
        Cell('b2', (-1, -1))
        Cell('b3', (0, -1))
        Cell('b4', (1, -1))
        Cell('b5', (2, -1))

        # Row "c"
        Cell('c1', (-2, 0))
        Cell('c2', (-1, 0))
        Cell('c3', (0, 0))
        Cell('c4', (1, 0))
        Cell('c5', (2, 0))

        # Row "d"
        Cell('d1', (-2, 1))
        Cell('d2', (-1, 1))
        Cell('d3', (0, 1))
        Cell('d4', (1, 1))
        Cell('d5', (2, 1))

        # Row "e"
        Cell('e1', (-2, 2))
        Cell('e2', (-1, 2))
        Cell('e3', (0, 2))
        Cell('e4', (1, 2))
        Cell('e5', (2, 2))

        Cell('E0', (-3, 3), reserve=True, player=Player.BLACK)
        Cell('E1', (-2, 3), reserve=True, player=Player.BLACK)
        Cell('E2', (-1, 3), reserve=True, player=Player.BLACK)
        Cell('E3', (0, 3), reserve=True, player=Player.BLACK)
        Cell('E4', (1, 3), reserve=True, player=Player.BLACK)
        Cell('E5', (2, 3), reserve=True, player=Player.BLACK)
        Cell('E6', (3, 3), reserve=True, player=Player.BLACK)


@enum.unique
class SimpleNotationCase(enum.Enum):

    INVALID = 'invalid'

    DROP_ONE_CUBE = 'x:xx'
    DROP_TWO_CUBES = 'x:xx/x:xx'

    MOVE_CUBE = 'xx-xx'
    MOVE_STACK = 'xx=xx'

    MOVE_CUBE_MOVE_STACK = 'xx-xx=xx'
    MOVE_STACK_MOVE_CUBE = 'xx=xx-xx'

    MOVE_CUBE_RELOCATE_KING = 'xx-xx/x:xx'
    MOVE_STACK_RELOCATE_KING = 'xx=xx/x:xx'

    MOVE_CUBE_MOVE_STACK_RELOCATE_KING = 'xx-xx=xx/x:xx'
    MOVE_STACK_MOVE_CUBE_RELOCATE_KING = 'xx=xx-xx/x:xx'


class Notation:

    __slots__ = ()


    def __init__(self):
        assert False


    @staticmethod
    def drop_cube(src_cube_label, dst_cell_name, previous_action=None):
        if previous_action is None:
            notation = ""
        else:
            notation = previous_action.notation + "/"
        notation += src_cube_label + ":" + dst_cell_name
        return notation


    @staticmethod
    def move_cube(src_cell_name, dst_cell_name, capture, previous_action=None):
        if previous_action is None:
            notation = src_cell_name + "-" + dst_cell_name
        else:
            notation = previous_action.notation + "-" + dst_cell_name

        if capture == Capture.NONE:
            pass

        elif capture in (Capture.SOME_CUBE, Capture.SOME_STACK):
            notation += "!"

        elif capture in (Capture.KING_CUBE, Capture.KING_STACK):
            notation += "!!"

        else:
            assert False

        return notation


    @staticmethod
    def move_stack(src_cell_name, dst_cell_name, capture, previous_action=None):
        if previous_action is None:
            notation = src_cell_name + "=" + dst_cell_name
        else:
            notation = previous_action.notation + "=" + dst_cell_name

        if capture == Capture.NONE:
            pass

        elif capture in (Capture.SOME_CUBE, Capture.SOME_STACK):
            notation += "!"

        elif capture in (Capture.KING_CUBE, Capture.KING_STACK):
            notation += "!!"

        else:
            assert False

        return notation


    @staticmethod
    def guess_symmetricals(notation):
        symmetricals = []

        if len(notation) == 9 and notation[1] == ':' and notation[6] == ':':
            # examples: w:a1/w:a2 | m:a1/m:a2
            # >>>>>>>>> 012345678
            cube_1 = notation[0]
            cube_2 = notation[5]
            cell_1 = notation[2:4]
            cell_2 = notation[7:9]

            if cube_1 == cube_2 and cell_1 != cell_2:
                symmetricals.append(cube_2 + ":" + cell_1 + "/" + cube_1 + ":" + cell_2)

        return symmetricals


    @staticmethod
    def relocate_king(src_king_label, dst_cell_name, previous_action=None):
        if previous_action is None:
            notation = ""
        else:
            notation = previous_action.notation + "/"
        notation += src_king_label + ":" + dst_cell_name
        return notation


    @staticmethod
    def simplify_notation(notation):
        return notation.strip().replace(' ', '').replace('!', '')


    @staticmethod
    def classify_notation(notation):
        notation_simplified = Notation.simplify_notation(notation)
        notation_case = Notation.classify_simple_notation(notation_simplified)

        # guess number of capture
        capture = 0
        if re.match(r"^.*!.*$ ", notation):
            capture += 1
            if re.match(r"^.*![^!]+!.*$ ", notation):
                capture += 1

        return (notation_case, capture)


    @staticmethod
    def classify_simple_notation(notation):
        if re.match(r'^([KFRPSMW]|[kfrpsmw]):[a-e][1-5]$', notation):
            # drop one cube
            return SimpleNotationCase.DROP_ONE_CUBE

        elif re.match(r'^([KFRPSMW]|[kfrpsmw]):[a-e][1-5]/([KFRPSMW]|[kfrpsmw]):[a-e][1-5]$', notation):
            # drop two cubes
            return SimpleNotationCase.DROP_TWO_CUBES

        elif re.match(r'^[a-e][1-5]-[a-e][1-5]$', notation):
            # move cube
            return SimpleNotationCase.MOVE_CUBE

        elif re.match(r'^[a-e][1-5]=[a-e][1-5]$', notation):
            # move stack
            return SimpleNotationCase.MOVE_STACK

        elif re.match(r'^[a-e][1-5]-[a-e][1-5]=[a-e][1-5]$', notation):
            # move cube move stack
            return SimpleNotationCase.MOVE_CUBE_MOVE_STACK

        elif re.match(r'^[a-e][1-5]=[a-e][1-5]-[a-e][1-5]$', notation):
            # move stack move cube
            return SimpleNotationCase.MOVE_STACK_MOVE_CUBE

        elif re.match(r'^[a-e][1-5]-[a-e][1-5]/[Kk]:[a-e][1-5]$', notation):
            # move cube relocate king
            return SimpleNotationCase.MOVE_CUBE_RELOCATE_KING

        elif re.match(r'^[a-e][1-5]=[a-e][1-5]/[Kk]:[a-e][1-5]$', notation):
            # move stack relocate king
            return SimpleNotationCase.MOVE_STACK_RELOCATE_KING

        elif re.match(r'^[a-e][1-5]-[a-e][1-5]=[a-e][1-5]/[Kk]:[a-e][1-5]$', notation):
            # move cube move stack relocate king
            return SimpleNotationCase.MOVE_CUBE_MOVE_STACK_RELOCATE_KING

        elif re.match(r'^[a-e][1-5]=[a-e][1-5]-[a-e][1-5]/[Kk]:[a-e][1-5]$', notation):
            # move stack move cube relocate king
            return SimpleNotationCase.MOVE_STACK_MOVE_CUBE_RELOCATE_KING

        else:
            return SimpleNotationCase.INVALID


    @staticmethod
    def validate_simple_notation(action_input, action_names):

        def split_actions(action_names):
            action_cases = {}

            for this_name in action_names:
                this_case = Notation.classify_simple_notation(this_name)
                if this_case not in action_cases:
                    action_cases[this_case] = set()
                action_cases[this_case].add(this_name)

            return action_cases


        action_input_simplified = Notation.simplify_notation(action_input)

        validated = action_input in action_names or action_input_simplified in action_names

        if validated:
            message = "validated action"

        else:
            action_cases = split_actions(action_names)
            action_input_case = Notation.classify_simple_notation(action_input_simplified)

            if action_input_case == SimpleNotationCase.INVALID:
                message = "invalid action syntax !"

            elif action_input_case not in action_cases:
                message = f"{action_input_case.value} : impossible action !"

            else:
                message = "invalid action !"

            # guess hints from each case of action
            action_hints = []
            for this_case in action_cases:
                # find the longest match from the start
                upper_length = min(len(action_input_simplified), len(this_case.value))
                match_length = 0
                for this_name in action_cases[this_case]:
                    for end in range(match_length, upper_length + 1):
                        if action_input_simplified[:end] == this_name[:end]:
                            match_length = max(match_length, end)
                        else:
                            break

                action_hints.append(action_input_simplified[:match_length] + this_case.value[match_length:])

            if len(action_hints) == 1:
                message += " hint : " + action_hints[0]
            else:
                message += " hints : " + "  ".join(action_hints)

        return (validated, message)


class JersiAction:

    __slots__ = ('notation', 'state', 'king_captures', 'some_captures')


    def __init__(self, notation, state, capture=Capture.NONE, previous_action=None):
        self.notation = notation
        self.state = state
        self.king_captures = set()
        self.some_captures = set()

        if previous_action is not None:
            self.king_captures.update(previous_action.king_captures)
            self.some_captures.update(previous_action.some_captures)

        if capture in (Capture.KING_CUBE, Capture.KING_STACK):
            self.king_captures.add(capture)

        elif capture in (Capture.SOME_CUBE, Capture.SOME_STACK):
            self.some_captures.add(capture)

        else:
            assert capture == Capture.NONE


    def __eq__(self, other):
        return (self.__class__ == other.__class__ and
                id(self.state) == id(other.state) and
                self.notation == other.notation)


    def __hash__(self):
        return hash((id(self.state), self.notation))


    def __repr__(self):
        return str(self)


    def __str__(self):
        return self.notation



class JersiActionAppender:

    __slots__ = ('__actions', '__notations')


    def __init__(self):
        self.__actions = []
        self.__notations = set()


    def append(self, action):
        if action.notation in self.__notations:
            return

        for symmetrical in Notation.guess_symmetricals(action.notation):
            if symmetrical in self.__notations:
                return

        self.__actions.append(action)
        self.__notations.add(action.notation)


    def get_actions(self):
        return self.__actions


class JersiState:

    __max_credit = 40
    __king_end_distances = None
    __center_cell_indices = None
    __reserve_cell_by_cube = None
    __prison_cell_by_cube = None

    __slots__ = ('__play_reserve', '__cube_status', '__cell_bottom', '__cell_top',
                 '__credit', '__player', '__turn',
                 '__actions', '__actions_by_simple_names', '__actions_by_names',
                 '__taken', '__terminal_case', '__terminated', '__rewards')


    def __init__(self, play_reserve=True):
        
        self.__play_reserve = play_reserve

        self.__cube_status = None
        self.__cell_bottom = None
        self.__cell_top = None

        self.__credit = JersiState.__max_credit
        self.__player = Player.WHITE
        self.__turn = 1

        self.__actions = None
        self.__actions_by_simple_names = None
        self.__actions_by_names = None
        self.__taken = False
        self.__terminal_case = None
        self.__terminated = None
        self.__rewards = None

        self.__init_cell_top_and_bottom()
        self.__init_cube_status()
        self.__init_king_end_distances()
        self.__init_center_cell_indices()
        self.__init_reserve_cell_by_cube()
        self.__init_prison_cell_by_cube()


    def __fork(self):
        
        state = copy.copy(self)

        state.__cube_status = copy.deepcopy(state.__cube_status)
        state.__cell_bottom = copy.deepcopy(state.__cell_bottom)
        state.__cell_top = copy.deepcopy(state.__cell_top)

        state.__actions = None
        state.__actions_by_simple_names = None
        state.__actions_by_names = None
        state.__taken = False
        state.__terminal_case = None
        state.__terminated = None
        state.__rewards = None

        return state


    def __init_cube_status(self):

        self.__cube_status = array.array('b', [CubeStatus.ACTIVATED for _ in Cube.all])

        for (cube_index, cube) in enumerate(Cube.all):

            if cube.sort in (CubeSort.MOUNTAIN, CubeSort.WISE):
                if self.__play_reserve:
                    self.__cube_status[cube_index] = CubeStatus.RESERVED
                else:
                    self.__cube_status[cube_index] = CubeStatus.UNUSED

            if not (cube_index in self.__cell_bottom or cube_index in self.__cell_top):
                self.__cube_status[cube_index] = CubeStatus.UNUSED


    def __init_cell_top_and_bottom(self):

        self.__cell_top = array.array('b', [Null.CUBE for _ in Cell.all])
        self.__cell_bottom = array.array('b', [Null.CUBE for _ in Cell.all])

        # whites
        self.__set_cube_at_cell_by_names('R1', 'a1')
        self.__set_cube_at_cell_by_names('P1', 'a2')
        self.__set_cube_at_cell_by_names('S1', 'a3')
        self.__set_cube_at_cell_by_names('F1', 'a4')
        self.__set_cube_at_cell_by_names('K1', 'a5')


        # blacks
        self.__set_cube_at_cell_by_names('r1', 'e5')
        self.__set_cube_at_cell_by_names('p1', 'e4')
        self.__set_cube_at_cell_by_names('s1', 'e3')
        self.__set_cube_at_cell_by_names('f1', 'e2')
        self.__set_cube_at_cell_by_names('k1', 'e1')

        if self.__play_reserve:
            # white reserve
            self.__set_cube_at_cell_by_names('W1', 'A5')
            self.__set_cube_at_cell_by_names('M1', 'A6')

            # black reserve
            self.__set_cube_at_cell_by_names('w1', 'E5')
            self.__set_cube_at_cell_by_names('m1', 'E6')


    def __init_king_end_distances(self):

        if JersiState.__king_end_distances is None:

            JersiState.__king_end_distances = [array.array('b', [0 for _ in Cell.all]) for _ in Player]

            for player in Player:

                for king_cell in Cell.all:

                    king_cell_index = king_cell.index
                    king_position_uv = king_cell.position_uv

                    king_distance = math.inf
                    for cell_index in Cell.get_king_end_indices(player):
                        cell_position_uv = Cell.all[cell_index].position_uv
                        king_distance = min(king_distance,
                                            cell_distance(king_position_uv, cell_position_uv))

                    JersiState.__king_end_distances[player][king_cell_index] = int(math.ceil(king_distance))


    def __init_center_cell_indices(self):

        if JersiState.__center_cell_indices is None:

            center_names = ['b2', 'b3', 'b4',
                            'c2', 'c3', 'c4',
                            'd2', 'd3', 'd4']

            JersiState.__center_cell_indices = array.array('b',
                                                         [Cell.get(name).index for name in center_names])


    def __init_reserve_cell_by_cube(self):
        
        
        def define_reserve_cell(cube_name, cell_name):
            cube_index = Cube.get(cube_name).index
            cell_index = Cell.get(cell_name).index
            JersiState.__reserve_cell_by_cube[cube_index] = cell_index
            
        
        if JersiState.__reserve_cell_by_cube is None:
            JersiState.__reserve_cell_by_cube = array.array('b', [Null.CELL for _ in Cube.all])

            # whites
            define_reserve_cell('R1', 'A0')
            define_reserve_cell('P1', 'A1')
            define_reserve_cell('S1', 'A2')
            define_reserve_cell('F1', 'A3')
            define_reserve_cell('K1', 'A4')
    
            # blacks
            define_reserve_cell('r1', 'E0')
            define_reserve_cell('p1', 'E1')
            define_reserve_cell('s1', 'E2')
            define_reserve_cell('f1', 'E3')
            define_reserve_cell('k1', 'E4')
    
            # white reserve
            define_reserve_cell('W1', 'A5')
            define_reserve_cell('M1', 'A6')

            # black reserve
            define_reserve_cell('w1', 'E5')
            define_reserve_cell('m1', 'E6')
        

    def __init_prison_cell_by_cube(self):
        
        
        def define_prison_cell(cube_name, cell_name):
            cube_index = Cube.get(cube_name).index
            cell_index = Cell.get(cell_name).index
            JersiState.__prison_cell_by_cube[cube_index] = cell_index
            
        
        if JersiState.__prison_cell_by_cube is None:
            JersiState.__prison_cell_by_cube = array.array('b', [Null.CELL for _ in Cube.all])

            # whites
            define_prison_cell('R1', 'E0')
            define_prison_cell('P1', 'E1')
            define_prison_cell('S1', 'E2')
            define_prison_cell('F1', 'E3')
            define_prison_cell('K1', 'E4')
    
            # blacks
            define_prison_cell('r1', 'A0')
            define_prison_cell('p1', 'A1')
            define_prison_cell('s1', 'A2')
            define_prison_cell('f1', 'A3')
            define_prison_cell('k1', 'A4')
    
            # white prison
            define_prison_cell('W1', 'E5')

            # black prison
            define_prison_cell('w1', 'A5')
        

    def __set_cube_at_cell_by_names(self, cube_name, cell_name):
        cube_index = Cube.get(cube_name).index
        cell_index = Cell.get(cell_name).index
        self.__set_cube_at_cell(cube_index, cell_index)


    def __set_cube_at_cell(self, cube_index, cell_index):
        
        cell = Cell.all[cell_index]
        
        if cell.reserve:
            cube = Cube.all[cube_index]
            
            if cell.player == Player.WHITE:
                
                if cube.player == Player.WHITE:
                    assert self.__cell_top[cell_index] == Null.CUBE
                    self.__cell_top[cell_index] = cube_index            
                
                elif cube.player == Player.BLACK:
                    assert self.__cell_bottom[cell_index] == Null.CUBE
                    self.__cell_bottom[cell_index] = cube_index            
                
            
            elif cell.player == Player.BLACK:
                
                if cube.player == Player.BLACK:
                    assert self.__cell_bottom[cell_index] == Null.CUBE
                    self.__cell_bottom[cell_index] = cube_index            
                
                elif cube.player == Player.WHITE:
                    assert self.__cell_top[cell_index] == Null.CUBE
                    self.__cell_top[cell_index] = cube_index  
            
            else:
                assert cell.player is None

        else:

            if self.__cell_bottom[cell_index] == Null.CUBE:
                # cell has zero cube
                self.__cell_bottom[cell_index] = cube_index
    
            elif self.__cell_top[cell_index] == Null.CUBE:
                # cell has one cube
                self.__cell_top[cell_index] = cube_index
    
            else:
                # cell is expected with either zero or one cube
                assert False


    def __set_cube_at_reserve(self, cube_index):
        cell_index = JersiState.__reserve_cell_by_cube[cube_index]
        self.__set_cube_at_cell(cube_index, cell_index)


    def __set_cube_at_prison(self, cube_index):
        cell_index = JersiState.__prison_cell_by_cube[cube_index]
        self.__set_cube_at_cell(cube_index, cell_index)


    def __manage_new_prisoner(self, cube_index):
        assert self.__cube_status[cube_index] == CubeStatus.CAPTURED

        opposite_index = Cube.get_opposite_index(cube_index)
        
        if not self.__play_reserve or self.__cube_status[opposite_index] != CubeStatus.CAPTURED:
            self.__set_cube_at_prison(cube_index)
            
        else:
            # exchange the prisoners that return to their reserves
            
            opposite_cell_index = JersiState.__prison_cell_by_cube[opposite_index]
            
            if self.__cell_top[opposite_cell_index] == opposite_index:
                self.__cell_top[opposite_cell_index] = Null.CUBE
                
            elif self.__cell_bottom[opposite_cell_index] == opposite_index:
                self.__cell_bottom[opposite_cell_index] = Null.CUBE
                
            else:
                assert False

            self.__cube_status[cube_index] = CubeStatus.RESERVED            
            self.__set_cube_at_reserve(cube_index)

            self.__cube_status[opposite_index] = CubeStatus.RESERVED            
            self.__set_cube_at_reserve(opposite_index)


    def show(self):

        shift = " " * len("a1KR")

        print()

        for (row_shift_count, row_cell_names) in Cell.get_layout():

            row_text = shift*row_shift_count

            for cell_name in row_cell_names:

                row_text += cell_name
                cell = Cell.get(cell_name)
                cell_index = cell.index

                top_index = self.__cell_top[cell_index]
                bottom_index = self.__cell_bottom[cell_index]

                if bottom_index == Null.CUBE:
                    row_text += ".."

                elif top_index == Null.CUBE:
                    bottom_label = Cube.all[bottom_index].label
                    row_text += "." + bottom_label

                elif top_index != Null.CUBE:
                    top_label = Cube.all[top_index].label
                    bottom_label = Cube.all[bottom_index].label
                    row_text += top_label + bottom_label

                else:
                    assert False

                row_text += shift
            print(row_text)


        print()
        print(self.get_summary())


    def get_king_end_distances(self):
        """Distance to end cells of kings"""

        king_distances = [0 for _ in Player]

        for player in Player:

            king_index= Cube.get_king_index(player)

            if king_index in self.__cell_bottom:
                king_cell_index = self.__cell_bottom.index(king_index)
            else:
                king_cell_index = self.__cell_top.index(king_index)

            king_distances[player] = JersiState.__king_end_distances[player][king_cell_index]

        return king_distances


    def get_center_cell_indices(self):
        return JersiState.__center_cell_indices


    def get_capture_counts(self):
        counts = [0 for _ in Player]

        for (cube_index, cube_status) in enumerate(self.__cube_status):

            if cube_status == CubeStatus.CAPTURED:
                cube = Cube.all[cube_index]
                counts[cube.player] += 1

        return counts


    def get_fighter_counts(self):
        counts = [0 for _ in Player]

        for (cube_index, cube_status) in enumerate(self.__cube_status):

            if cube_status == CubeStatus.ACTIVATED:
                cube = Cube.all[cube_index]
                if cube.sort in (CubeSort.FOOL, CubeSort.PAPER, CubeSort.ROCK, CubeSort.SCISSORS):
                    counts[cube.player] += 1

        return counts


    def get_reserve_counts(self):
        counts = [0 for _ in Player]

        for (cube_index, cube_status) in enumerate(self.__cube_status):

            if cube_status == CubeStatus.RESERVED:
                cube = Cube.all[cube_index]
                counts[cube.player] += 1

        return counts


    def get_summary(self):

        reserved_labels = collections.Counter()
        captured_labels = collections.Counter()

        for (cube_index, cube_status) in enumerate(self.__cube_status):
            cube = Cube.all[cube_index]

            if cube_status == CubeStatus.RESERVED:
                reserved_labels[cube.label] += 1

            elif cube_status == CubeStatus.CAPTURED:
                captured_labels[cube.label] += 1

        summary = (
            f"turn {self.__turn} / player {Player.name(self.__player)} / credit {self.__credit} / " +
             "reserved %s" % " ".join([f"{label}:{count}" for (label, count) in sorted(reserved_labels.items())]) + " / " +
             "captured %s" % " ".join([f"{label}:{count}" for (label, count) in sorted(captured_labels.items())]))

        return summary


    @staticmethod
    def get_max_credit():
        return JersiState.__max_credit


    @staticmethod
    def set_max_credit(max_credit):
        assert max_credit > 0
        JersiState.__max_credit = max_credit


    def get_current_player(self):
        return self.__player


    def get_cell_bottom(self):
        return self.__cell_bottom


    def get_cell_top(self):
        return self.__cell_top


    def get_other_player(self):

        if self.__player == Player.WHITE:
            return Player.BLACK

        elif self.__player == Player.BLACK:
            return Player.WHITE

        else:
            assert False


    def get_rewards(self):
        assert self.is_terminal()
        return self.__rewards


    def get_terminal_case(self):
        return self.__terminal_case


    def get_turn(self):
        return self.__turn


    def take_action(self, action):

        state = action.state
        
        if state.__taken == False:
            state.__taken = True
            state.__player = state.get_other_player()
            state.__turn += 1
            state.__credit = max(0, state.__credit - 1)

            if len(action.some_captures) != 0:
                state.__credit = JersiState.__max_credit

            elif len(action.king_captures) != 0 and action.king_captures != set([Capture.KING_CUBE]):
                state.__credit = JersiState.__max_credit

        return state


    def take_action_by_simple_name(self, action_name):
       assert action_name in self.get_action_simple_names()
       action = self.__actions_by_simple_names[action_name]
       self.take_action(action)


    def take_action_by_name(self, action_name):
       assert action_name in self.get_action_names()
       action = self.__actions_by_names[action_name]
       self.take_action(action)


    def is_terminal(self):

        if self.__terminated is None:

            self.__terminated = False

            white_captured = self.__cube_status[Cube.white_king_index] == CubeStatus.CAPTURED
            black_captured = self.__cube_status[Cube.black_king_index] == CubeStatus.CAPTURED

            white_arrived = False
            black_arrived = False

            if not (white_captured or black_captured):

                if Cube.white_king_index in self.__cell_bottom:
                    cell_index = self.__cell_bottom.index(Cube.white_king_index)
                    white_arrived = cell_index in Cell.get_king_end_indices(Player.WHITE)

                else:
                    cell_index = self.__cell_top.index(Cube.white_king_index)
                    white_arrived = cell_index in Cell.get_king_end_indices(Player.WHITE)

                if not white_arrived:

                    if Cube.black_king_index in self.__cell_bottom:
                        cell_index = self.__cell_bottom.index(Cube.black_king_index)
                        black_arrived = cell_index in Cell.get_king_end_indices(Player.BLACK)

                    else:
                        cell_index = self.__cell_top.index(Cube.black_king_index)
                        black_arrived = cell_index in Cell.get_king_end_indices(Player.BLACK)

            if white_captured:
                # white king captured without possible relocation ==> black wins
                self.__terminated = True
                self.__terminal_case = TerminalCase.WHITE_CAPTURED
                self.__rewards = [Reward.DRAW for _ in Player]
                self.__rewards[Player.BLACK] = Reward.WIN
                self.__rewards[Player.WHITE] = Reward.LOSS

            elif black_captured:
                # black king captured without possible relocation ==> white wins
                self.__terminated = True
                self.__terminal_case = TerminalCase.BLACK_CAPTURED
                self.__rewards = [Reward.DRAW for _ in Player]
                self.__rewards[Player.WHITE] = Reward.WIN
                self.__rewards[Player.BLACK] = Reward.LOSS

            elif white_arrived:
                # white arrived at goal ==> white wins
                self.__terminated = True
                self.__terminal_case = TerminalCase.WHITE_ARRIVED
                self.__rewards = [Reward.DRAW for _ in Player]
                self.__rewards[Player.WHITE] = Reward.WIN
                self.__rewards[Player.BLACK] = Reward.LOSS

            elif black_arrived:
                # black arrived at goal ==> black wins
                self.__terminated = True
                self.__terminal_case = TerminalCase.BLACK_ARRIVED
                self.__rewards = [Reward.DRAW for _ in Player]
                self.__rewards[Player.BLACK] = Reward.WIN
                self.__rewards[Player.WHITE] = Reward.LOSS

            elif self.__credit == 0:
                # credit is exhausted ==> nobody wins
                self.__terminated = True
                self.__terminal_case = TerminalCase.ZERO_CREDIT
                self.__rewards = [Reward.DRAW for _ in Player]

            elif not self.has_action():
                # the current player looses and the other player wins
                self.__terminated = True
                self.__rewards = [Reward.DRAW for _ in Player]

                if self.__player == Player.WHITE:
                    self.__terminal_case = TerminalCase.WHITE_BLOCKED
                    self.__rewards[Player.WHITE] = Reward.LOSS
                    self.__rewards[Player.BLACK] = Reward.WIN
                else:
                    self.__terminal_case = TerminalCase.BLACK_BLOCKED
                    self.__rewards[Player.BLACK] = Reward.LOSS
                    self.__rewards[Player.WHITE] = Reward.WIN

        return self.__terminated


    def get_actions(self, shuffle=True):
        if self.__actions is None:
            self.__actions = self.__find_moves() + self.__find_drops()
            # Better to shuffle actions here than by MCTS searcher for example
            if shuffle:
                random.shuffle(self.__actions)
        return self.__actions


    def has_action(self):

        moves = self.__find_cube_first_moves(find_one=True)
        if len(moves) != 0:
            king_relocation_moves = self.__find_king_relocations(moves)
            if len(king_relocation_moves) != 0:
                return True

        drops = self.__find_drops(find_one=True)
        if len(drops) != 0:
            return True

        if len(moves) != 0 and len(king_relocation_moves) == 0:
            return len(self.get_actions(shuffle=False)) != 0
        else:
            return False


    def __create_action_by_names(self):
        self.__actions_by_names = {}
        self.__actions_by_simple_names = {}

        for action in self.get_actions():
            self.__actions_by_names[action.notation] = action

            action_name = Notation.simplify_notation(action.notation)
            self.__actions_by_simple_names[action_name] = action


    def get_action_names(self):
        if self.__actions_by_names is None:
            self.__create_action_by_names()
        return list(sorted(self.__actions_by_names.keys()))


    def get_action_simple_names(self):
        if self.__actions_by_simple_names is None:
            self.__create_action_by_names()
        return list(sorted(self.__actions_by_simple_names.keys()))


    def get_action_by_name(self, action_name):
       assert action_name in self.get_action_names()
       action = self.__actions_by_names[action_name]
       return action


    def get_action_by_simple_name(self, action_name):
       assert action_name in self.get_action_simple_names()
       action = self.__actions_by_simple_names[action_name]
       return action


    ### Action finders

    def __find_drops(self, find_one=False):

        action_appender = JersiActionAppender()
        found_one = False

        for cube_1_index in self.__find_droppable_cubes():
            if find_one and found_one:
                break

            for destination_1 in Cell.get_all_active_indices():
                action_1 = self.__try_drop(cube_1_index, destination_1)
                if action_1 is not None:
                    action_appender.append(action_1)
                    if find_one:
                        found_one = True
                        break

                    state_1 = action_1.state.__fork()

                    for cube_2_index in state_1.__find_droppable_cubes():
                        for destination_2 in [destination_1] + Cell.get_next_fst_active_indices(destination_1):
                            action_2 = state_1.__try_drop(cube_2_index, destination_2, previous_action=action_1)
                            if action_2 is not None:
                                action_appender.append(action_2)

        return action_appender.get_actions()


    def __find_moves(self):
        actions = self.__find_stack_first_moves() + self.__find_cube_first_moves()
        return self.__find_king_relocations(actions)


    def __find_king_relocations(self, move_actions):

        actions = []

        king_index = Cube.get_king_index(self.get_other_player())
        king = Cube.all[king_index]

        for action in move_actions:
            if len(action.king_captures) != 0:
                can_relocate_king = False

                for destination_king in Cell.get_king_begin_indices(king.player):
                    action_king = action.state.__try_relocate_king(king_index, destination_king, previous_action=action)
                    if action_king is not None:
                        actions.append(action_king)
                        can_relocate_king = True

                if not can_relocate_king:
                    actions.append(action)

            else:
                actions.append(action)

        return actions


    def __find_cube_first_moves(self, find_one=False):
        actions = []
        found_one = False

        for source_1 in self.__find_cells_with_movable_cube():

            if find_one and found_one:
                break

            for direction_1 in CellDirection:
                destination_1 = Cell.get_next_fst_indices(source_1, direction_1)
                if destination_1 != Null.CELL:
                    action_1 = self.__try_move_cube(source_1, destination_1)
                    if action_1 is not None:
                        actions.append(action_1)
                        if find_one:
                            found_one = True
                            break

                        state_1 = action_1.state.__fork()
                        if state_1.__is_cell_with_movable_stack(destination_1):

                            for direction_2 in CellDirection:
                                destination_21 = Cell.get_next_fst_indices(destination_1, direction_2)
                                if destination_21 != Null.CELL:
                                    action_21 = state_1.__try_move_stack(destination_1, destination_21, previous_action=action_1)
                                    if action_21 is not None:
                                        actions.append(action_21)

                                    if state_1.__cell_bottom[destination_21] == Null.CUBE:
                                        # stack can cross destination_21 with zero cube
                                        destination_22 = Cell.get_next_snd_indices(destination_1, direction_2)
                                        if destination_22 != Null.CELL:
                                            action_22 = state_1.__try_move_stack(destination_1, destination_22, previous_action=action_1)
                                            if action_22 is not None:
                                                actions.append(action_22)
        return actions


    def __find_stack_first_moves(self, find_one=False):

        actions = []
        found_one = False

        for source_1 in self.__find_cells_with_movable_stack():

            if find_one and found_one:
                break

            for direction_1 in CellDirection:
                destination_11 = Cell.get_next_fst_indices(source_1, direction_1)
                if destination_11 != Null.CELL:
                    action_11 = self.__try_move_stack(source_1, destination_11)
                    if action_11 is not None:
                        actions.append(action_11)
                        if find_one:
                            found_one = True
                            break

                        state_11 = action_11.state.__fork()

                        for direction_21 in CellDirection:
                            destination_21 = Cell.get_next_fst_indices(destination_11, direction_21)
                            if destination_21 != Null.CELL:
                                action_21 = state_11.__try_move_cube(destination_11, destination_21, previous_action=action_11)
                                if action_21 is not None:
                                    actions.append(action_21)

                    if self.__cell_bottom[destination_11] == Null.CUBE:
                        # stack can cross destination_11 with zero cube
                        destination_12 = Cell.get_next_snd_indices(source_1, direction_1)
                        if destination_12 != Null.CELL:
                            action_12 = self.__try_move_stack(source_1, destination_12)
                            if action_12 is not None:
                                actions.append(action_12)

                                state_12 = action_12.state.__fork()

                                for direction_22 in CellDirection:
                                    destination_22 = Cell.get_next_fst_indices(destination_12, direction_22)
                                    if destination_22 != Null.CELL:
                                        action_22 = state_12.__try_move_cube(destination_12, destination_22, previous_action=action_12)
                                        if action_22 is not None:
                                            actions.append(action_22)
        return actions

    ### Cubes and cells finders

    def __find_droppable_cubes(self):
        droppable_cubes = []

        for (src_cube_index, src_cube_status) in enumerate(self.__cube_status):
            if src_cube_status == CubeStatus.RESERVED:
                cube = Cube.all[src_cube_index]
                if cube.player == self.__player:
                    droppable_cubes.append(src_cube_index)

        return droppable_cubes


    def __find_cells_with_movable_cube(self):
         return [x for x in Cell.get_all_active_indices() if self.__is_cell_with_movable_cube(x)]


    def __find_cells_with_movable_stack(self):
        return [x for x in Cell.get_all_active_indices() if self.__is_cell_with_movable_stack(x)]

    ### Cell predicates

    def __is_cell_with_movable_cube(self, cell_index):
        to_be_returned = False

        if Cell.all[cell_index].reserve:
            to_be_returned = False

        elif self.__cell_top[cell_index] != Null.CUBE:
            cube_index = self.__cell_top[cell_index]
            cube = Cube.all[cube_index]
            if cube.player == self.__player and cube.sort != CubeSort.MOUNTAIN:
                to_be_returned = True

        elif self.__cell_bottom[cell_index] != Null.CUBE:
            cube_index = self.__cell_bottom[cell_index]
            cube = Cube.all[cube_index]
            if cube.player == self.__player and cube.sort != CubeSort.MOUNTAIN:
                to_be_returned = True

        return to_be_returned


    def __is_cell_with_movable_stack(self, cell_index):
        to_be_returned = False

        if Cell.all[cell_index].reserve:
            to_be_returned = False

        else:
            top_index = self.__cell_top[cell_index]
            bottom_index = self.__cell_bottom[cell_index]

            if top_index != Null.CUBE and bottom_index != Null.CUBE:

                top = Cube.all[top_index]
                bottom = Cube.all[bottom_index]

                if (top.player == self.__player and bottom.player == self.__player and
                    top.sort != CubeSort.MOUNTAIN and bottom.sort != CubeSort.MOUNTAIN):
                    to_be_returned = True

        return to_be_returned

    ### Action triers

    def __try_drop(self, src_cube_index, dst_cell_index, previous_action=None):

        src_cube = Cube.all[src_cube_index]
        src_cube_label = src_cube.label
        dst_cell_name = Cell.all[dst_cell_index].name
        notation = Notation.drop_cube(src_cube_label, dst_cell_name, previous_action=previous_action)

        if src_cube.player != self.__player:
            action = None

        elif self.__cube_status[src_cube_index] != CubeStatus.RESERVED:
            action = None

        elif Cell.all[dst_cell_index].reserve:
            action = None

        elif self.__cell_bottom[dst_cell_index] == Null.CUBE:
            # destination cell has zero cube

            state = self.__fork()

            if src_cube_index in state.__cell_top:
                src_cell_index = state.__cell_top.index(src_cube_index)
                state.__cell_top[src_cell_index] = Null.CUBE
            else:
                src_cell_index = state.__cell_bottom.index(src_cube_index)
                state.__cell_bottom[src_cell_index] = Null.CUBE

            assert Cell.all[src_cell_index].reserve

            state.__cell_bottom[dst_cell_index] = src_cube_index
            state.__cube_status[src_cube_index] = CubeStatus.ACTIVATED
            action = JersiAction(notation, state)

        elif self.__cell_top[dst_cell_index] == Null.CUBE:
            # destination cell has one cube

            dst_bottom_index = self.__cell_bottom[dst_cell_index]
            dst_bottom = Cube.all[dst_bottom_index]

            if dst_bottom.player != self.__player:
                action = None

            elif dst_bottom.sort == CubeSort.KING:
                action = None

            elif src_cube.sort == CubeSort.MOUNTAIN and dst_bottom.sort != CubeSort.MOUNTAIN:
                action = None

            else:
                state = self.__fork()

                if src_cube_index in state.__cell_top:
                    src_cell_index = state.__cell_top.index(src_cube_index)
                    state.__cell_top[src_cell_index] = Null.CUBE
                else:
                    src_cell_index = state.__cell_bottom.index(src_cube_index)
                    state.__cell_bottom[src_cell_index] = Null.CUBE

                assert Cell.all[src_cell_index].reserve

                state.__cell_top[dst_cell_index] = src_cube_index
                state.__cube_status[src_cube_index] = CubeStatus.ACTIVATED
                action = JersiAction(notation, state)

        else:
            # destination cell has two cubes
            action = None

        return action


    def __try_relocate_king(self, king_index, dst_cell_index, previous_action=None):

        king = Cube.all[king_index]
        king_label = king.label
        dst_cell_name = Cell.all[dst_cell_index].name

        if king.sort != CubeSort.KING:
            action = None

        elif king.player == self.__player:
            action = None

        elif self.__cube_status[king_index] != CubeStatus.CAPTURED:
            action = None

        elif dst_cell_index not in Cell.get_king_begin_indices(king.player):
            action = None

        elif self.__cell_top[dst_cell_index] != Null.CUBE:
            # cell has two cubes
            action = None

        elif self.__cell_bottom[dst_cell_index] == Null.CUBE:
            # cell has zero cube

            state = self.__fork()
            state.__cell_bottom[dst_cell_index] = king_index
            state.__cube_status[king_index] = CubeStatus.ACTIVATED
            notation = Notation.relocate_king(king_label, dst_cell_name, previous_action=previous_action)
            action = JersiAction(notation, state, capture=Capture.KING_CUBE, previous_action=previous_action)

        else:
            # cell has one cube

            dst_bottom_index = self.__cell_bottom[dst_cell_index]
            dst_bottom = Cube.all[dst_bottom_index]

            if dst_bottom.player == king.player or dst_bottom.sort == CubeSort.MOUNTAIN:

                state = self.__fork()
                state.__cell_top[dst_cell_index] = king_index
                state.__cube_status[king_index] = CubeStatus.ACTIVATED
                notation = Notation.relocate_king(king_label, dst_cell_name, previous_action=previous_action)
                action = JersiAction(notation, state, capture=Capture.KING_CUBE, previous_action=previous_action)

            else:
                action = None

        return action


    def __try_move_cube(self, src_cell_index, dst_cell_index, previous_action=None):

        src_cell_name = Cell.all[src_cell_index].name
        dst_cell_name = Cell.all[dst_cell_index].name

        if not self.__is_cell_with_movable_cube(src_cell_index):
            action = None

        elif Cell.all[dst_cell_index].reserve:
            action = None

        elif self.__cell_bottom[dst_cell_index] == Null.CUBE:
            # destination cell has zero cube

            state = self.__fork()

            if state.__cell_top[src_cell_index] != Null.CUBE:
                src_cube_index = state.__cell_top[src_cell_index]
                state.__cell_top[src_cell_index] = Null.CUBE
            else:
                src_cube_index = state.__cell_bottom[src_cell_index]
                state.__cell_bottom[src_cell_index] = Null.CUBE
            state.__cell_bottom[dst_cell_index] = src_cube_index

            notation = Notation.move_cube(src_cell_name, dst_cell_name, capture=Capture.NONE, previous_action=previous_action)
            action = JersiAction(notation, state, previous_action=previous_action)

        elif self.__cell_top[dst_cell_index] == Null.CUBE:
            # destination cell has one cube

            dst_bottom_index = self.__cell_bottom[dst_cell_index]
            dst_bottom = Cube.all[dst_bottom_index]

            if self.__cell_top[src_cell_index] != Null.CUBE:
                src_cube_index = self.__cell_top[src_cell_index]
            else:
                src_cube_index = self.__cell_bottom[src_cell_index]
            src_cube = Cube.all[src_cube_index]

            if dst_bottom.sort == CubeSort.MOUNTAIN:
                state = self.__fork()

                if state.__cell_top[src_cell_index] != Null.CUBE:
                    state.__cell_top[src_cell_index] = Null.CUBE
                else:
                    state.__cell_bottom[src_cell_index] = Null.CUBE
                state.__cell_top[dst_cell_index] = src_cube_index

                notation = Notation.move_cube(src_cell_name, dst_cell_name, capture=Capture.NONE, previous_action=previous_action)
                action = JersiAction(notation, state, previous_action=previous_action)

            elif dst_bottom.player != self.__player:

                if src_cube.beats(dst_bottom):
                    # Capture the bottom cube

                    state = self.__fork()

                    state.__cell_bottom[dst_cell_index] = Null.CUBE
                    state.__cube_status[dst_bottom_index] = CubeStatus.CAPTURED

                    if dst_bottom.sort == CubeSort.KING:
                        capture = Capture.KING_CUBE
                    else:
                        capture = Capture.SOME_CUBE

                    if state.__cell_top[src_cell_index] != Null.CUBE:
                        state.__cell_top[src_cell_index] = Null.CUBE
                    else:
                        state.__cell_bottom[src_cell_index] = Null.CUBE
                    state.__cell_bottom[dst_cell_index] = src_cube_index
                    
                    if capture == Capture.SOME_CUBE:
                        state.__manage_new_prisoner(dst_bottom_index)

                    notation = Notation.move_cube(src_cell_name, dst_cell_name, capture=capture, previous_action=previous_action)
                    action = JersiAction(notation, state, capture=capture, previous_action=previous_action)
                else:
                    action = None

            elif dst_bottom.sort == CubeSort.KING:
                action = None

            else:
                state = self.__fork()

                if state.__cell_top[src_cell_index] != Null.CUBE:
                    state.__cell_top[src_cell_index] = Null.CUBE
                else:
                    state.__cell_bottom[src_cell_index] = Null.CUBE
                state.__cell_top[dst_cell_index] = src_cube_index

                notation = Notation.move_cube(src_cell_name, dst_cell_name, capture=Capture.NONE, previous_action=previous_action)
                action = JersiAction(notation, state, previous_action=previous_action)

        else:
            # destination cell has two cubes
            dst_top_index = self.__cell_top[dst_cell_index]
            dst_bottom_index = self.__cell_bottom[dst_cell_index]

            dst_top = Cube.all[dst_top_index]
            dst_bottom = Cube.all[dst_bottom_index]

            if self.__cell_top[src_cell_index] != Null.CUBE:
                src_cube_index = self.__cell_top[src_cell_index]
            else:
                src_cube_index = self.__cell_bottom[src_cell_index]
            src_cube = Cube.all[src_cube_index]

            if dst_top.player == self.__player:
                action = None

            elif src_cube.beats(dst_top) and dst_bottom.sort == CubeSort.MOUNTAIN:
                # Capture the top of the stack
                state = self.__fork()

                state.__cell_top[dst_cell_index] = Null.CUBE
                state.__cube_status[dst_top_index] = CubeStatus.CAPTURED

                if dst_top.sort == CubeSort.KING:
                    capture = Capture.KING_CUBE
                else:
                    capture = Capture.SOME_CUBE

                if state.__cell_top[src_cell_index] != Null.CUBE:
                    state.__cell_top[src_cell_index] = Null.CUBE
                else:
                    state.__cell_bottom[src_cell_index] = Null.CUBE
                state.__cell_top[dst_cell_index] = src_cube_index
                    
                if capture == Capture.SOME_CUBE:
                    state.__manage_new_prisoner(dst_top_index)

                notation = Notation.move_cube(src_cell_name, dst_cell_name, capture=capture, previous_action=previous_action)
                action = JersiAction(notation, state, capture=capture, previous_action=previous_action)

            elif src_cube.beats(dst_top) and dst_bottom.sort != CubeSort.MOUNTAIN:
                # Capture the stack
                state = self.__fork()

                state.__cell_top[dst_cell_index] = Null.CUBE
                state.__cell_bottom[dst_cell_index] = Null.CUBE

                state.__cube_status[dst_top_index] = CubeStatus.CAPTURED
                state.__cube_status[dst_bottom_index] = CubeStatus.CAPTURED

                if dst_top.sort == CubeSort.KING:
                    capture = Capture.KING_STACK
                else:
                    capture = Capture.SOME_STACK

                if state.__cell_top[src_cell_index] != Null.CUBE:
                    state.__cell_top[src_cell_index] = Null.CUBE
                else:
                    state.__cell_bottom[src_cell_index] = Null.CUBE
                state.__cell_bottom[dst_cell_index] = src_cube_index
                    
                if capture == Capture.SOME_STACK:
                    state.__manage_new_prisoner(dst_top_index)
                    state.__manage_new_prisoner(dst_bottom_index)
                else:
                    state.__manage_new_prisoner(dst_bottom_index)

                notation = Notation.move_cube(src_cell_name, dst_cell_name, capture=capture, previous_action=previous_action)
                action = JersiAction(notation, state, capture=capture, previous_action=previous_action)

            else:
                action = None

        return action


    def __try_move_stack(self, src_cell_index, dst_cell_index, previous_action=None):

        src_cell_name = Cell.all[src_cell_index].name
        dst_cell_name = Cell.all[dst_cell_index].name

        if not self.__is_cell_with_movable_cube(src_cell_index):
            action = None

        elif Cell.all[dst_cell_index].reserve:
            action = None

        elif self.__cell_bottom[dst_cell_index] == Null.CUBE:
            # destination cell has zero cube

            state = self.__fork()

            src_bottom_index = state.__cell_bottom[src_cell_index]
            src_top_index = state.__cell_top[src_cell_index]

            state.__cell_bottom[src_cell_index] = Null.CUBE
            state.__cell_top[src_cell_index] = Null.CUBE

            state.__cell_bottom[dst_cell_index] = src_bottom_index
            state.__cell_top[dst_cell_index] = src_top_index

            notation = Notation.move_stack(src_cell_name, dst_cell_name, capture=Capture.NONE, previous_action=previous_action)
            action = JersiAction(notation, state, previous_action=previous_action)

        elif self.__cell_top[dst_cell_index] == Null.CUBE:
            # destination cell has one cube

            src_bottom_index = self.__cell_bottom[src_cell_index]
            src_top_index = self.__cell_top[src_cell_index]

            src_top = Cube.all[src_top_index]

            dst_bottom_index = self.__cell_bottom[dst_cell_index]
            dst_bottom = Cube.all[dst_bottom_index]

            if src_top.player == dst_bottom.player:
                action = None

            elif src_top.beats(dst_bottom):
                # capture the bottom cube
                state = self.__fork()

                state.__cell_bottom[dst_cell_index] = Null.CUBE
                state.__cube_status[dst_bottom_index] = CubeStatus.CAPTURED

                if dst_bottom.sort == CubeSort.KING:
                    capture = Capture.KING_CUBE
                else:
                    capture = Capture.SOME_CUBE

                state.__cell_bottom[src_cell_index] = Null.CUBE
                state.__cell_top[src_cell_index] = Null.CUBE

                state.__cell_bottom[dst_cell_index] = src_bottom_index
                state.__cell_top[dst_cell_index] = src_top_index

                if capture == Capture.SOME_CUBE:
                    state.__manage_new_prisoner(dst_bottom_index)

                notation = Notation.move_stack(src_cell_name, dst_cell_name, capture=capture, previous_action=previous_action)
                action = JersiAction(notation, state, capture=capture, previous_action=previous_action)

            else:
                action = None

        else:
            # destination cell has two cubes

            src_top_index = self.__cell_top[src_cell_index]
            src_top = Cube.all[src_top_index]

            src_bottom_index = self.__cell_bottom[src_cell_index]

            dst_top_index = self.__cell_top[dst_cell_index]
            dst_top = Cube.all[dst_top_index]

            dst_bottom_index = self.__cell_bottom[dst_cell_index]
            dst_bottom = Cube.all[dst_bottom_index]

            if src_top.player == dst_top.player:
                action = None

            elif src_top.beats(dst_top) and dst_bottom.sort != CubeSort.MOUNTAIN:
                # capture the stack
                state = self.__fork()

                state.__cell_bottom[dst_cell_index] = Null.CUBE
                state.__cell_top[dst_cell_index] = Null.CUBE

                state.__cube_status[dst_bottom_index] = CubeStatus.CAPTURED
                state.__cube_status[dst_top_index] = CubeStatus.CAPTURED

                if dst_top.sort == CubeSort.KING:
                    capture = Capture.KING_STACK
                else:
                    capture = Capture.SOME_STACK

                state.__cell_bottom[src_cell_index] = Null.CUBE
                state.__cell_top[src_cell_index] = Null.CUBE

                state.__cell_bottom[dst_cell_index] = src_bottom_index
                state.__cell_top[dst_cell_index] = src_top_index

                if capture == Capture.SOME_STACK:
                    state.__manage_new_prisoner(dst_top_index)
                    state.__manage_new_prisoner(dst_bottom_index)
                else:
                    state.__manage_new_prisoner(dst_bottom_index)

                notation = Notation.move_stack(src_cell_name, dst_cell_name, capture=capture, previous_action=previous_action)
                action = JersiAction(notation, state, capture=capture, previous_action=previous_action)

            else:
                action = None

        return action


class MctsState:
    """Adaptater to mcts.StateInterface for JersiState"""

    __slots__ = ('__mikjersi_state', '__maximizer_player')


    def __init__(self, mikjersi_state, maximizer_player):
        self.__mikjersi_state = mikjersi_state
        self.__maximizer_player = maximizer_player


    def get_mikjersi_state(self):
        return self.__mikjersi_state


    def getCurrentPlayer(self):
       """ Returns 1 if it is the maximizer player's turn to choose an action,
       or -1 for the minimiser player"""
       return 1 if self.__mikjersi_state.get_current_player() == self.__maximizer_player else -1


    def isTerminal(self):
        return self.__mikjersi_state.is_terminal()


    def getReward(self):
        """Returns the reward for this state: 0 for a draw,
        positive for a win by maximizer player or negative for a win by the minimizer player.
        Only needed for terminal states."""

        mikjersi_rewards = self.__mikjersi_state.get_rewards()

        if mikjersi_rewards[self.__maximizer_player] == Reward.DRAW:
            mcts_reward = 0

        elif mikjersi_rewards[self.__maximizer_player] == Reward.WIN:
            mcts_reward = 1

        else:
            mcts_reward = -1

        return mcts_reward


    def getPossibleActions(self):
        return self.__mikjersi_state.get_actions()


    def takeAction(self, action):
        return MctsState(self.__mikjersi_state.take_action(action), self.__maximizer_player)


class MinimaxState:

    __slots__ = ('__mikjersi_state', '__maximizer_player')


    def __init__(self, mikjersi_state, maximizer_player):
        self.__mikjersi_state = mikjersi_state
        self.__maximizer_player = maximizer_player


    def get_mikjersi_state(self):
        return self.__mikjersi_state


    def get_current_maximizer_player(self):
        return self.__maximizer_player


    def is_terminal(self):
        return self.__mikjersi_state.is_terminal()


    def get_reward(self):
        """Returns the reward for this state: 0 for a draw,
        positive for a win by maximizer player or negative for a win by the minimizer player.
        Only needed for terminal states."""

        mikjersi_rewards = self.__mikjersi_state.get_rewards()

        if mikjersi_rewards[self.__maximizer_player] == Reward.DRAW:
            minimax_reward = 0

        elif mikjersi_rewards[self.__maximizer_player] == Reward.WIN:
            minimax_reward = 1

        else:
            minimax_reward = -1

        return minimax_reward


    def get_actions(self, shuffle):
        return self.__mikjersi_state.get_actions(shuffle)


    def take_action(self, action):
        return MinimaxState(self.__mikjersi_state.take_action(action), self.__maximizer_player)


def extractStatistics(mcts_searcher, action):
    statistics = {}
    statistics['rootNumVisits'] = mcts_searcher.root.numVisits
    statistics['rootTotalReward'] = mcts_searcher.root.totalReward
    statistics['actionNumVisits'] = mcts_searcher.root.children[action].numVisits
    statistics['actionTotalReward'] = mcts_searcher.root.children[action].totalReward
    return statistics


def mikjersiSelectAction(action_names):


    def score_move_name(move_name):

        catpures = re.sub(r"[^!]", "",move_name)
        catpures = re.sub(r"!+", "100",catpures)

        stacks = re.sub(r"[^=]", "",move_name).replace("=", "10")

        cubes = re.sub(r"[^-]", "",move_name).replace("-", "1")

        move_score = 0

        if catpures != "":
            move_score += float(catpures)

        if stacks != "":
            move_score += float(stacks)

        if cubes != "":
            move_score += float(cubes)

        return move_score


    assert len(action_names) != 0
    (drop_names, move_names) = partition(lambda x: re.match(r"^.*[-=].*$", str(x)), action_names)

    drop_names = list(drop_names)
    move_names = list(move_names)
    assert len(drop_names) + len(move_names) != 0

    drop_probability = 0.25

    if len(drop_names) != 0 and random.random() <= drop_probability:
        action_name = random.choice(drop_names)

    elif len(move_names) != 0:
        move_weights = list(map(score_move_name, move_names))
        assert len(move_weights) != 0
        action_name = random.choices(move_names, weights=move_weights, k=1)[0]
    
    else:
        action_name = random.choice(drop_names)
        

    return action_name


def mikjersiRandomPolicy(state):
    while not state.isTerminal():
        try:
            mikjersi_state = state.get_mikjersi_state()

            action_names = mikjersi_state.get_action_names()
            action_name = mikjersiSelectAction(action_names)
            action = mikjersi_state.get_action_by_name(action_name)

        except IndexError:
            raise Exception("Non-terminal state has no possible actions: " + str(state))
        state = state.takeAction(action)
    return state.getReward()


class HumanSearcher():

    __slots__ = ('__name', '__action_simple_name', '__use_command_line')


    def __init__(self, name):
        self.__name = name
        self.__action_simple_name = None
        self.__use_command_line = False


    def get_name(self):
        return self.__name


    def is_interactive(self):
        return True


    def use_command_line(self, condition):
        assert condition in (True, False)
        self.__use_command_line = condition


    def set_action_simple_name(self, action_name):
        assert not self.__use_command_line
        self.__action_simple_name = action_name


    def search(self, state):

        if self.__use_command_line:
            return self.__search_using_command_line(state)

        else:
            action = state.get_action_by_simple_name(self.__action_simple_name)
            self.__action_simple_name = None
            return action


    def __search_using_command_line(self, state):
        assert self.__use_command_line

        action_names = state.get_action_simple_names()

        action_validated = False
        while not action_validated:
            action_input = Notation.simplify_notation(input("HumanSearcher: action? "))
            (action_validated, validation_message) = Notation.validate_simple_notation(action_input, action_names)
            print(validation_message)

        action = state.get_action_by_simple_name(action_input)

        print(f"HumanSearcher: action {action} has been selected")

        return action


class RandomSearcher():

    __slots__ = ('__name')


    def __init__(self, name):
        self.__name = name


    def get_name(self):
        return self.__name


    def is_interactive(self):
        return False


    def search(self, state):
        actions = state.get_actions()

        (drop_actions, move_actions) = partition(lambda x: re.match(r"^.*[-=].*$", str(x)), actions)
        drop_actions = list(drop_actions)
        move_actions = list(move_actions)

        if len(move_actions) == 0:
            action = random.choice(drop_actions)

        else:
            drop_probability = 0.25

            if len(drop_actions) != 0 and random.random() <= drop_probability:
                action = random.choice(drop_actions)
            else:
                action = random.choice(move_actions)

        return action


class MinimaxSearcher():

    __slots__ = ('__name', '__max_depth', '__max_children',
                 '__distance_weight', '__capture_weight', 
                 '__fighter_weight', '__reserve_weight',
                 '__center_weight',
                 '__debug')


    default_weights_by_depth = dict()

    default_weights_by_depth[1] = {'distance_weight':16,
                                   'capture_weight':8,
                                   'fighter_weight':4,
                                   'center_weight':2,
                                   'reserve_weight':1}

    default_weights_by_depth[2] = {'distance_weight':16,
                                   'capture_weight':8,
                                   'fighter_weight':4,
                                   'center_weight':2,
                                   'reserve_weight':1}


    def __init__(self, name, max_depth=1, max_children=None,
                  distance_weight=None, capture_weight=None,
                  fighter_weight=None, reserve_weight=None,
                  center_weight=None):

        self.__debug = False

        assert max_depth >= 1

        if max_depth in MinimaxSearcher.default_weights_by_depth:
            default_weights = MinimaxSearcher.default_weights_by_depth[max_depth]
        else:
            default_weights = MinimaxSearcher.default_weights_by_depth[2]


        self.__name = name
        self.__max_depth = max_depth
        self.__max_children = max_children


        if distance_weight is not None:
            self.__distance_weight = distance_weight
        else:
            self.__distance_weight = default_weights['distance_weight']


        if capture_weight is not None:
            self.__capture_weight = capture_weight
        else:
            self.__capture_weight = default_weights['capture_weight']


        if fighter_weight is not None:
            self.__fighter_weight = fighter_weight
        else:
            self.__fighter_weight = default_weights['fighter_weight']


        if reserve_weight is not None:
            self.__reserve_weight = reserve_weight
        else:
            self.__reserve_weight = default_weights['reserve_weight']


        if center_weight is not None:
            self.__center_weight = center_weight
        else:
            self.__center_weight = default_weights['center_weight']


    def get_name(self):
        return self.__name


    def is_interactive(self):
        return False


    def search(self, state):
        do_check = False
        
        initial_state = MinimaxState(state, state.get_current_player())

        (best_value, action_values) = self.alphabeta(state=initial_state,
                                                    player=1,
                                                    return_action_values=True)
 
        if do_check:       
             self.check(initial_state, best_value, action_values)
            
        if self.__debug:
            print()
            
        best_actions = list()
        for (action, action_value) in action_values.items():
            if action_value == best_value:
                best_actions.append(action)
                if self.__debug:
                    print("MinimaxSearcher.search: best (action, value)=",(action, action_value))               

        print()
        print("%d best_actions with best value %.1f" % (len(best_actions), best_value))

        action = random.choice(best_actions)

        return action


    def check(self, initial_state, best_value, action_values):

        (best_value_ref, action_values_ref) = self.minimax(state=initial_state,
                                                    player=1,
                                                    return_action_values=True)

        if self.__debug:
            print()
            print("MinimaxSearcher.check: best_value_ref=",best_value_ref)
            print("MinimaxSearcher.check: best_value=",best_value)

        if self.__debug:
            print()
            
        best_actions_ref = list()
        for (action_ref, action_value_ref) in action_values_ref.items():
            if action_value_ref == best_value_ref:
                best_actions_ref.append(action_ref)
                if self.__debug:
                    print("MinimaxSearcher.check: best (action_ref, action_value_ref)=", 
                          (action_ref, action_value_ref))               

        if self.__debug:
            print()
            print("%d best_actions_ref with best value %.1f" % (len(best_actions_ref), best_value_ref))
           
        best_actions = list()
        for (action, action_value) in action_values.items():
            if action_value == best_value:
                best_actions.append(action)
                if self.__debug:
                    print("MinimaxSearcher.check: best (action, action_value)=", 
                          (action, action_value))               
 
        if self.__debug:
            print()
        
        action_names_ref = set(map(str, action_values_ref.keys()))
        action_names = set(map(str, action_values.keys()))

        best_names_ref = set(best_actions_ref)
        best_names = set(best_actions)

        assert best_value == best_value_ref
        
        assert len(action_names) <= len(action_names_ref)
        assert len(action_names - action_names_ref) == 0
 
        assert len(best_names) <= len(best_names_ref)
        assert len(best_names - best_names_ref) == 0
    

    def state_value(self, state, depth):
        # evaluate favorability for mikjersi_maximizer_player

        assert depth >= 0

        # evaluate as if mikjersi_maximizer_player == Player.WHITE and use minimax_maximizer_sign
        mikjersi_maximizer_player = state.get_current_maximizer_player()
        
        if mikjersi_maximizer_player == Player.WHITE:
            minimax_maximizer_sign = 1
        else:
            minimax_maximizer_sign = -1

        value = 0

        mikjersi_state = state.get_mikjersi_state()

        if mikjersi_state.is_terminal():
            # >> amplify terminal value using the depth (rationale: winning faster is safer)
            
            white_reward = mikjersi_state.get_rewards()[Player.WHITE]

            if white_reward == Reward.WIN:
                value = minimax_maximizer_sign*OMEGA_2*(depth + 1)

            elif white_reward == Reward.DRAW:
                # >> no use of minimax_maximizer_sign because the DRAW applies to both WHITE and BLACK
                value = OMEGA*(depth + 1)

            else:
                value = minimax_maximizer_sign*(-OMEGA_2)*(depth + 1)

        else:

            # white and black distances to their goals or ends
            king_distances = mikjersi_state.get_king_end_distances()
            distance_difference = minimax_maximizer_sign*(king_distances[Player.BLACK] - king_distances[Player.WHITE])

            # white and black with captured status
            capture_counts = mikjersi_state.get_capture_counts()
            capture_difference = minimax_maximizer_sign*(capture_counts[Player.BLACK] - capture_counts[Player.WHITE])

            # white and black with active fighters status
            fighter_counts = mikjersi_state.get_fighter_counts()
            fighter_difference = minimax_maximizer_sign*(fighter_counts[Player.WHITE] - fighter_counts[Player.BLACK])

            # white and black with reserved status
            reserve_counts = mikjersi_state.get_reserve_counts()     
            reserve_difference = minimax_maximizer_sign*(reserve_counts[Player.WHITE] - reserve_counts[Player.BLACK])

            # white and black fighter cubes in the central zone
            white_center_count = 0
            black_center_count = 0

            cell_bottom = mikjersi_state.get_cell_bottom()
            cell_top= mikjersi_state.get_cell_top()

            for cell_index in mikjersi_state.get_center_cell_indices():
                for cube_index in [cell_bottom[cell_index], cell_top[cell_index]]:
                    if cube_index != Null.CUBE:
                        cube = Cube.all[cube_index]

                        if cube.sort in (CubeSort.FOOL, CubeSort.PAPER, CubeSort.ROCK, CubeSort.SCISSORS):
                            if cube.player == Player.WHITE:
                                white_center_count += 1

                            elif cube.player == Player.BLACK:
                                black_center_count += 1
                    else:
                        break

            center_difference = minimax_maximizer_sign*(white_center_count - black_center_count)

            # normalize each feature in the intervall [-1, +1]

            distance_norm = 4
            capture_norm = 5
            fighter_norm = 4
            reserve_norm = 6
            center_norm = 9
            
            assert distance_difference <= distance_norm
            assert -distance_difference <= distance_norm
            
            assert capture_difference <= capture_norm
            assert -capture_difference <= capture_norm
            
            assert fighter_difference <= fighter_norm
            assert -fighter_difference <= fighter_norm
            
            assert center_difference <= center_norm
            assert -center_difference <= center_norm

            distance_difference = distance_difference/distance_norm
            capture_difference = capture_difference/capture_norm
            fighter_difference = fighter_difference/fighter_norm
            reserve_difference = reserve_difference/reserve_norm
            center_difference = center_difference/center_norm
            
            # synthesis

            value += self.__distance_weight*distance_difference
            value += self.__capture_weight*capture_difference
            value += self.__fighter_weight*fighter_difference
            value += self.__reserve_weight*reserve_difference
            value += self.__center_weight*center_difference

        return value


    def reduce_actions(self, actions):

        if (self.__max_children is not None and len(actions) > self.__max_children):
            if self.__debug:
                print("--- reduce actions")           

            (drop_actions, move_actions) = partition(lambda x: re.match(r"^.*[-=].*$", str(x)), actions)
            drop_actions = list(drop_actions)
            move_actions = list(move_actions)

            if len(move_actions) > self.__max_children:
                # sample the move actions according to their destination cells
                move_actions.sort(key=lambda x: re.sub(r"/[kK]:..$", "", str(x)).replace("!","")[-2:])

                selected_move_actions = list()
                for action_chunk in chunks(move_actions, self.__max_children):
                    selected_move_actions.append(random.choice(action_chunk))

                move_actions = selected_move_actions

            if len(drop_actions) != 0:
                drop_count = self.__max_children - len(move_actions)

                # >> let us admit some tolerance regarding the __max_children criterion
                # >> by adding a small fraction of drop actions
                drop_probability = 0.25
                drop_count = max(drop_count, int(math.ceil(drop_probability*len(move_actions))))

                drop_actions = random.choices(drop_actions, k=drop_count)
                actions = move_actions + drop_actions
            else:
                actions = move_actions

        return actions


    def sort_actions(self, actions):

        def score_action(action):
            captures = re.sub(r"[^!]", "", str(action))
            return len(captures)
        
        if self.__debug:
            print("--- sort actions")           
        actions.sort(key=score_action, reverse=True)


    def minimax(self, state, player, depth=None, return_action_values=False):

        if depth is None:
            depth =self.__max_depth


        if depth == 0 or state.is_terminal():
            state_value = self.state_value(state, depth)
            
            if self.__debug:
                print()
                print("minimax at depth %d evaluates leaf state %d with value %f" % 
                      (depth, id(state),  state_value))
                
            return state_value

        if self.__debug:
            print()
            print("minimax at depth %d evaluates state %d ..." % (depth, id(state)))           

        assert player == -1 or player == 1
        
        if player == -1:
            assert not return_action_values

        if return_action_values:
            action_values = dict()

        actions = state.get_actions(shuffle=False)
        
        if player == 1:
            
            state_value = -math.inf
            
            for action in actions:
                child_state = state.take_action(action)
                
                child_value = self.minimax(state=child_state, player=-player, depth=depth - 1)
    
                if return_action_values:
                    action_values[action] = child_value
                    
                state_value = max(state_value, child_value)    

        elif player == -1:
            
            state_value = math.inf
            
            for action in actions:
                child_state = state.take_action(action)
                
                child_value = self.minimax(state=child_state, player=-player, depth=depth - 1)
                    
                state_value = min(state_value, child_value)    
                
        if self.__debug:
            print()
            print("minimax at depth %d evaluates state %d with value %f" % (depth, id(state), state_value))           
     
        if return_action_values:
            return (state_value, action_values)
        else:
            return state_value


    def alphabeta(self, state, player, depth=None, alpha=None, beta=None, return_action_values=False):

        use_sort = True

        if alpha is None:
            alpha = -math.inf

        if beta is None:
            beta = math.inf

        if depth is None:
            depth = self.__max_depth
            
        assert alpha <= beta

        if depth == 0 or state.is_terminal():
            state_value = self.state_value(state, depth)
            
            if self.__debug:
                print()
                print("alphabeta at depth %d evaluates leaf state %d with value %f" % 
                      (depth, id(state),  state_value))
                
            return state_value

        if return_action_values:
            action_values = dict()

        if self.__debug:
            print()
            print("alphabeta at depth %d evaluates state %d ..." % (depth, id(state)))           

        assert player == -1 or player == 1
        
        if player == -1:
            assert not return_action_values

        if return_action_values:
            action_values = dict()

        actions = state.get_actions(shuffle=False)
        actions = self.reduce_actions(actions)
        if use_sort:
            self.sort_actions(actions)
        
        if player == 1:
            
            state_value = -math.inf
            
            for action in actions:
                child_state = state.take_action(action)
                
                child_value = self.alphabeta(state=child_state, player=-player, depth=depth - 1,
                                              alpha=alpha, beta=beta)
    
                if return_action_values:
                    action_values[action] = child_value
                    
                state_value = max(state_value, child_value)    
                
                if state_value >= beta:
                    if self.__debug:
                        print("--- beta cut-off")
                    break
                        
                alpha = max(alpha, state_value)    

        elif player == -1:
            
            state_value = math.inf
            
            for (action_index, action) in enumerate(actions):
                child_state = state.take_action(action)
                
                child_value = self.alphabeta(state=child_state, player=-player, depth=depth - 1,
                                              alpha=alpha, beta=beta)
    
                state_value = min(state_value, child_value)    
                
                if state_value <= alpha:
                    if self.__debug:
                        print("--- alpha cut-off")
                    
                    if depth == (self.__max_depth - 1):
                        if state_value == alpha and action_index != (len(actions) - 1):
                            # >> prevent final return of actions with falsely equal values due to cut-off
                            # >> rationale: without cut-off it could be that state_value < alpha 
                            state_value -= 1/OMEGA
                            assert state_value < alpha
                            if self.__debug:
                                print("--- force state_value < alpha")
                    
                    break
                        
                beta = min(beta, state_value)    

        if self.__debug:
            print()
            print("alphabeta at depth %d evaluates state %d with value %f" % (depth, id(state), state_value))           
     
        if return_action_values:
            return (state_value, action_values)
        else:
            return state_value


    def negamax(self, state, player, depth=None, return_action_values=False):

        if depth is None:
            depth =self.__max_depth

        if depth == 0 or state.is_terminal():
            state_value = player*self.state_value(state, depth)
            
            if self.__debug:
                print()
                print("negamax at depth %d evaluates leaf state %d with value %f" % 
                      (depth, id(state),  state_value))
                
            return state_value

        if return_action_values:
            action_values = dict()

        actions = state.get_actions(shuffle=False)

        if self.__debug:
            print()
            print("negamax at depth %d evaluates state %d ..." % (depth, id(state)))           
            
        state_value = -math.inf
        
        for action in actions:
            child_state = state.take_action(action)
            
            child_value = -self.negamax(state=child_state, player=-player, depth=depth - 1)

            if return_action_values:
                action_values[action] = child_value
                
            state_value = max(state_value, child_value)    

        if self.__debug:
            print()
            print("negamax at depth %d evaluates state %d with value %f" % (depth, id(state), state_value))           
     
        if return_action_values:
            return (state_value, action_values)
        else:
            return state_value


class MctsSearcher():

    __slots__ = ('__name', '__time_limit', '__iteration_limit', '__capture_weight', '__searcher')


    def __init__(self, name, time_limit=None, iteration_limit=None, rolloutPolicy=mcts.randomPolicy):
        self.__name = name

        default_time_limit = 1_000

        assert time_limit is None or iteration_limit is None

        if time_limit is None and iteration_limit is None:
            time_limit = default_time_limit

        self.__time_limit = time_limit
        self.__iteration_limit = iteration_limit


        if self.__time_limit is not None:
            # time in milli-seconds
            self.__searcher = JersiMcts(timeLimit=self.__time_limit, rolloutPolicy=rolloutPolicy)

        elif self.__iteration_limit is not None:
            # number of mcts rounds
            self.__searcher = JersiMcts(iterationLimit=self.__iteration_limit, rolloutPolicy=rolloutPolicy)


    def get_name(self):
        return self.__name


    def is_interactive(self):
        return False


    def search(self, state):

        # >> when search is done, ignore the automatically selected action
        _ = self.__searcher.search(initialState=MctsState(state, state.get_current_player()))

        # heuristic: amonst best actions forget drop-actions i.e. selection a move action when possible

        best_actions = self.__searcher.getBestActions()
        best_move_actions = list(filter(lambda x: re.match(r"^.*[-=].*$", str(x)), best_actions))
        if len(best_move_actions) != 0:
            print("forget %d best drop actions !" % (len(best_actions) - len(best_move_actions)))
            best_actions = best_move_actions

        action = random.choice(best_actions)

        statistics = extractStatistics(self.__searcher, action)
        print("mcts statitics:" +
              f" chosen action= {statistics['actionTotalReward']} total reward" +
              f" over {statistics['actionNumVisits']} visits /"
              f" all explored actions= {statistics['rootTotalReward']} total reward" +
              f" over {statistics['rootNumVisits']} visits")

        if _do_debug:
            for (child_action, child) in self.__searcher.root.children.items():
                print(f"    action {child_action} numVisits={child.numVisits} totalReward={child.totalReward}")

        return action


class SearcherCatalog:

    __slots__ = ('__catalog')


    def __init__(self):
        self.__catalog = {}


    def add(self, searcher):
        searcher_name = searcher.get_name()
        assert searcher_name not in self.__catalog
        self.__catalog[searcher_name] = searcher


    def get_names(self):
        return list(sorted(self.__catalog.keys()))


    def get(self, name):
        assert name in self.__catalog
        return self.__catalog[name]


SEARCHER_CATALOG = SearcherCatalog()

SEARCHER_CATALOG.add( HumanSearcher("human") )
SEARCHER_CATALOG.add( RandomSearcher("random") )

SEARCHER_CATALOG.add( MinimaxSearcher("minimax1", max_depth=1) )
SEARCHER_CATALOG.add( MinimaxSearcher("minimax2", max_depth=2) )
SEARCHER_CATALOG.add( MinimaxSearcher("minimax3", max_depth=3) )
SEARCHER_CATALOG.add( MinimaxSearcher("minimax4", max_depth=4) )

SEARCHER_CATALOG.add( MctsSearcher("mcts-30s-jrp", time_limit=30_000, rolloutPolicy=mikjersiRandomPolicy) )
SEARCHER_CATALOG.add( MctsSearcher("mcts-60s-jrp", time_limit=60_000, rolloutPolicy=mikjersiRandomPolicy) )
SEARCHER_CATALOG.add( MctsSearcher("mcts-90s-jrp", time_limit=90_000, rolloutPolicy=mikjersiRandomPolicy) )


class Game:

    __slots__ = ('__searcher', '__mikjersi_state', '__log', '__turn', '__last_action', '__turn_duration')


    def __init__(self):
        self.__searcher = [None, None]

        self.__mikjersi_state = None
        self.__log = None
        self.__turn = None
        self.__last_action = None
        self.__turn_duration = {Player.WHITE:[], Player.BLACK:[]}


    def set_white_searcher(self, searcher):
        self.__searcher[Player.WHITE] = searcher


    def set_black_searcher(self, searcher):
        self.__searcher[Player.BLACK] = searcher


    def start(self, play_reserve=True):

        assert self.__searcher[Player.WHITE] is not None
        assert self.__searcher[Player.BLACK] is not None

        self.__mikjersi_state = JersiState(play_reserve)

        self.__mikjersi_state.show()

        self.__log = "Game started"


    def get_log(self):
        return self.__log


    def get_turn(self):
        assert self.__turn is not None
        return self.__turn


    def get_last_action(self):
        assert self.__last_action is not None
        return self.__last_action


    def get_summary(self):
        return self.__mikjersi_state.get_summary()


    def get_state(self):
        return self.__mikjersi_state


    def get_rewards(self):
        return self.__mikjersi_state.get_rewards()


    def has_next_turn(self):
        return not self.__mikjersi_state.is_terminal()


    def next_turn(self):

        self.__log = ""

        if self.has_next_turn():
            player = self.__mikjersi_state.get_current_player()
            player_name = f"{Player.name(player)}-{self.__searcher[player].get_name()}"
            action_count = len(self.__mikjersi_state.get_actions())

            print()
            print(f"{player_name} is thinking ...")

            turn_start = time.time()
            action = self.__searcher[player].search(self.__mikjersi_state)
            turn_end = time.time()
            turn_duration = turn_end - turn_start
            self.__turn_duration[player].append(turn_duration)

            self.__last_action = str(action)

            print(f"{player_name} is done after %.1f seconds" % turn_duration)

            self.__turn = self.__mikjersi_state.get_turn()

            self.__log = f"turn {self.__turn} : after {turn_duration:.1f} seconds {player_name} selects {action} amongst {action_count} actions"
            print(self.__log)
            print("-"*40)

            self.__mikjersi_state = self.__mikjersi_state.take_action(action)
            self.__mikjersi_state.show()

        if self.__mikjersi_state.is_terminal():

            rewards = self.__mikjersi_state.get_rewards()
            player = self.__mikjersi_state.get_current_player()

            print()
            print("-"*40)

            white_time = sum(self.__turn_duration[Player.WHITE])
            black_time = sum(self.__turn_duration[Player.BLACK])

            white_player = f"{Player.name(Player.WHITE)}-{self.__searcher[Player.WHITE].get_name()}"
            black_player = f"{Player.name(Player.BLACK)}-{self.__searcher[Player.BLACK].get_name()}"

            if rewards[Player.WHITE] == rewards[Player.BLACK]:
                self.__log = f"nobody wins ; the game is a draw between {white_player} and {black_player} ; {white_time:.0f} versus {black_time:.0f} seconds"

            elif rewards[Player.WHITE] > rewards[Player.BLACK]:
                self.__log = f"{white_player} wins against {black_player} ; {white_time:.0f} versus {black_time:.0f} seconds"

            else:
                self.__log = f"{black_player} wins against {white_player} ; {black_time:.0f} versus {white_time:.0f} seconds"

            print(self.__log)



def test_game_between_random_players():

    print("=====================================")
    print(" test_game_between_random_players ...")
    print("=====================================")

    default_max_credit = JersiState.get_max_credit()
    JersiState.set_max_credit(10_000)

    game = Game()

    game.set_white_searcher(RandomSearcher("random"))
    game.set_black_searcher(RandomSearcher("random"))

    game.start()

    while game.has_next_turn():
        game.next_turn()

    JersiState.set_max_credit(default_max_credit)

    print("=====================================")
    print("test_game_between_random_players done")
    print("=====================================")


def test_game_between_mcts_players():

    print("==================================")
    print("test_game_between_mcts_players ...")
    print("==================================")

    default_max_credit = JersiState.get_max_credit()
    JersiState.set_max_credit(10)

    game = Game()

    game.set_white_searcher(MctsSearcher("mcts-10s", time_limit=10_000))
    game.set_black_searcher(MctsSearcher("mcts-10i", iteration_limit=10))

    game.start()

    while game.has_next_turn():
        game.next_turn()

    JersiState.set_max_credit(default_max_credit)

    print("===================================")
    print("test_game_between_mcts_players done")
    print("===================================")


def test_game_between_random_and_human_players():

    print("==============================================")
    print("test_game_between_random_and_human_players ...")
    print("==============================================")

    default_max_credit = JersiState.get_max_credit()
    JersiState.set_max_credit(10)

    game = Game()

    human_searcher = HumanSearcher("human")
    human_searcher.use_command_line(True)
    game.set_white_searcher(human_searcher)

    game.set_black_searcher(RandomSearcher("random"))

    game.start()

    while game.has_next_turn():
        game.next_turn()

    JersiState.set_max_credit(default_max_credit)

    print("===============================================")
    print("test_game_between_random_and_human_players done")
    print("===============================================")


def test_game_between_minimax_players():

    print("=====================================")
    print(" test_game_between_minimax_players ...")
    print("=====================================")


    searcher_dict = dict()

    capture_weight_list = [1200, 1400, 1600]
    center_weight_list = [0, 100, 200, 400]

    for capture_weight in capture_weight_list:
        for center_weight in center_weight_list:
            searcher_name = "minimax2-%d-%d" % (capture_weight, center_weight)
            searcher = MinimaxSearcher(searcher_name,
                                        max_depth=2,
                                        capture_weight=capture_weight,
                                        center_weight=center_weight)
            assert searcher_name not in searcher_dict
            searcher_dict[searcher_name] = searcher


    (capture_weight, center_weight) = (1200, 400)
    searcher_name = "minimax1-%d-%d" % (capture_weight, center_weight)
    searcher = MinimaxSearcher(searcher_name,
                                max_depth=1,
                                capture_weight=capture_weight,
                                center_weight=center_weight)
    searcher_dict[searcher_name] = searcher

    searcher_points = collections.Counter()

    game_count = 5

    for x_searcher in searcher_dict.values():
        for y_searcher in searcher_dict.values():
            if x_searcher is y_searcher:
                continue


            x_points = 0
            y_points = 0


            for game_index in range(game_count):


                game = Game()
                game.set_white_searcher(x_searcher)
                game.set_black_searcher(y_searcher)
                x_player = Player.WHITE
                y_player = Player.BLACK

                game.start(play_reserve=False)
                while game.has_next_turn():
                    print("--> " + x_searcher.get_name() + " versus " +
                                   y_searcher.get_name() +  " game_index: %d" % game_index)
                    game.next_turn()

                rewards = game.get_rewards()

                if rewards[x_player] == Reward.WIN:
                    x_points += 2

                elif rewards[x_player] == Reward.DRAW:
                    x_points += 1

                if rewards[y_player] == Reward.WIN:
                    y_points += 2

                elif rewards[y_player] == Reward.DRAW:
                    y_points += 1


            print("game_count:", game_count, "/ x_points:", x_points, "/ y_points:", y_points)

            searcher_points[x_searcher.get_name()] += x_points
            searcher_points[y_searcher.get_name()] += y_points


    print()
    for (searcher_name, points) in sorted(searcher_points.items()):
        print("searcher %s has %d points" %(searcher_name, points))

    print()
    searcher_count = len(searcher_dict)
    searcher_game_count = 2*(searcher_count - 1)*game_count
    print("number of searchers:", searcher_count)
    print("number of games per searcher:", searcher_game_count)
    print()
    for (searcher_name, points) in sorted(searcher_points.items()):
        print("searcher %s has %.3f average points per game" %(searcher_name, points/searcher_game_count))

    print("=====================================")
    print("test_game_between_minimax_players done")
    print("=====================================")


def main():
    print(f"Hello from {os.path.basename(__file__)} version {__version__}")
    print(_COPYRIGHT_AND_LICENSE)

    if True:
        test_game_between_random_players()

    if False:
        test_game_between_mcts_players()

    if False:
        test_game_between_random_and_human_players()

    if False:
        test_game_between_minimax_players()

    if True:
        print()
        _ = input("main: done ; press enter to terminate")

    print(f"Bye from {os.path.basename(__file__)} version {__version__}")


Cube.init()
Cell.init()


if __name__ == "__main__":
    main()
