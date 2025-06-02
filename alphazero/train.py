import time
import traceback

import torch
import torch.nn.functional as F
from torch import nn


def exception_handler(train_func):
    """ 异常处理装饰器 """

    def wrapper(train_pipe_line, *args, **kwargs):
        try:
            train_func(train_pipe_line)
        except BaseException as e:
            if not isinstance(e, KeyboardInterrupt):
                traceback.print_exc()

            t = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))
            train_pipe_line.save_model(f'last_policy_value_net_{t}', 'train_losses', 'games')

    return wrapper


class PolicyValueLoss(nn.Module):
    """ 根据 self-play 产生的 `z` 和 `π` 计算误差 """

    def __init__(self):
        super().__init__()

    def forward(self, p_hat, pi, value, z, verbose=False):
        """ 前馈

        Parameters
        ----------
        p_hat: Tensor of shape (N, board_len^2)
            对数动作概率向量

        pi: Tensor of shape (N, board_len^2)
            `mcts` 产生的动作概率向量

        value: Tensor of shape (N, )
            对每个局面的估值

        z: Tensor of shape (N, )
            最终的游戏结果相对每一个玩家的奖赏
        """
        value_loss = F.mse_loss(value, z)
        policy_loss = 10 -torch.sum(pi * p_hat, dim=1).mean() * 10
        loss = value_loss + policy_loss
        return loss
