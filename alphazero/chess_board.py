# coding: utf-8
from copy import deepcopy
from typing import Tuple, List, Any

import numpy as np
import torch
from numpy import ndarray

from ruspy_city import Game


class ChessBoard:
    """
    棋盘类，用于存储棋盘状态和落子，判断游戏是否结束等
    """

    Player_Blue = 0
    Player_Green = 1

    action_to_pos = {
        0: (-3, 0),
        1: (-2, -1), 2: (-2, 0), 3: (-2, 1),
        4: (-1, -2), 5: (-1, -1), 6: (-1, 0), 7: (-1, 1), 8: (-1, 2),
        9: (0, -3), 10: (0, -2), 11: (0, -1), 12: (0, 0), 13: (0, 1), 14: (0, 2), 15: (0, 3),
        16: (1, -2), 17: (1, -1), 18: (1, 0), 19: (1, 1), 20: (1, 2),
        21: (2, -1), 22: (2, 0), 23: (2, 1),
        24: (3, 0)
    }

    pos_to_action = {
        (-3, 0): 0,
        (-2, -1): 1, (-2, 0): 2, (-2, 1): 3,
        (-1, -2): 4, (-1, -1): 5, (-1, 0): 6, (-1, 1): 7, (-1, 2): 8,
        (0, -3): 9, (0, -2): 10, (0, -1): 11, (0, 0): 12, (0, 1): 13, (0, 2): 14, (0, 3): 15,
        (1, -2): 16, (1, -1): 17, (1, 0): 18, (1, 1): 19, (1, 2): 20,
        (2, -1): 21, (2, 0): 22, (2, 1): 23,
        (3, 0): 24
    }

    def __init__(self, board_len=7, n_feature_planes=5, game=None):
        """
        :param board_len: 棋盘边长
        :param n_feature_planes: 特征平面数
        """
        if game is None:
            self.inner = Game(board_len, board_len)
        else:
            self.inner = game
        self.board_len = board_len
        self.n_feature_planes = n_feature_planes

        # index 0  蓝色 位置
        #       1  蓝色 上一个位置
        #       2  蓝色 上上个位置
        #       3  绿色 位置
        #       4  绿色 上一个位置
        #       5  绿色 上上个位置
        #       6  横向墙 位置
        #       7  横向墙 上一个位置
        #       8  横向墙 上上个位置
        #       9  纵向墙 位置
        #       10 纵向墙 上一个位置
        #       11 纵向墙 上上个位置
        #       12 该谁走了 0 for 蓝色 ; 1 for 绿色

        self.step_count = 0

    def copy(self) -> 'ChessBoard':
        """ 复制棋盘 """
        game = self.inner.clone()
        return ChessBoard(self.board_len, self.n_feature_planes, game)

    def get_player_pos(self) -> List[tuple[int, int]]:
        """ 获取玩家位置 """
        return self.inner.get_player_pos()

    def clear_board(self):
        """ 清空棋盘 """
        self.inner.clear_board()

        self.step_count = 0

    def __getattribute__(self, name):
        if name == "state":
            inner = super().__getattribute__("inner")
            return np.array(inner.get_state_planes())
        if name == "available_actions":
            inner = super().__getattribute__("inner")
            return inner.get_available_actions()
        if name == "player_pos":
            inner = super().__getattribute__("inner")
            return inner.get_player_pos()
        return super().__getattribute__(name)
    
    def do_action(self, action: int):
        """
        执行动作
        :param update_available_actions:
        :param action: 动作，范围为 0 ~ 99
        :return:
        """
        result = self.inner.do_action(action, True)
        if not result:
            raise ValueError(f'Illegal action {action}')

        self.step_count += 1

    def is_game_over(self) -> Tuple[bool, int]:
        """
        判断游戏是否结束
        :return: （是否结束， 胜利者） 胜利者为 0 代表 X 胜利， 1 代表 O 胜利， None 代表平局
        """
        return self.inner.is_game_over()

    def is_game_over_(self) -> tuple[bool, list[list[int, int]], list[list[int, int]]] | tuple[bool, None, None]:
        """
        判断游戏是否结束
        :return: （是否结束， X玩家可到达的位置， O玩家可到达的位置）
        """

        return self.inner.is_game_over_()

    def get_feature_planes(self) -> torch.Tensor:
        """
        获取特征平面
        :return: torch.Tensor of shape (n_feature_planes, board_len, board_len)
        """

        return torch.tensor(self.state, dtype=torch.float)

    def get_available_actions(self) -> List[int]:
        """
        获取当前可用动作
        :return: 动作列表
        """
        return self.inner.get_available_actions()

    @staticmethod
    def array_to_coordinates(array: np.ndarray) -> np.ndarray:
        """
        找出2D数组中所有为1的位置
        :param array: 2维数组
        :return: 1的位置二维数组，注意即使只有一个1，也是数组
        """
        array = array.copy()

        return np.stack(np.where(array == 1)).T.tolist()

    @staticmethod
    def array_to_only_coordinate(array: np.ndarray) -> tuple[int, int]:
        """
        找出2D数组中唯一的1的位置
        :param array: 2维数组
        :return: 1的位置
        """
        return int(np.where(array == 1)[0][0]), int(np.where(array == 1)[1][0])

    def coordinates_to_array(self, coordinates: list) -> np.ndarray:
        """
        根据坐标列表生成2D数组
        :param coordinates: 坐标列表
        :return: 2D数组，只有坐标列表中的位置为1，其余为0
        """
        array = np.zeros(shape=(self.board_len, self.board_len))
        for coordinate in coordinates:
            array[coordinate[0]][coordinate[1]] = 1
        return array
