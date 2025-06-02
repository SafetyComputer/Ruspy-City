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


