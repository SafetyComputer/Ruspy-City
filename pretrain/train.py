from collections import deque

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np
import sys

from alphazero import PolicyValueNet
from alphazero.train import PolicyValueLoss

class GameDataset(Dataset):
    """ 自我博弈数据集类，每个样本为元组 `(feature_planes, pi, z)` """

    def __init__(self, data_list):
        super().__init__()
        self.__data_deque = deque(data_list)

    def __len__(self):
        return len(self.__data_deque)

    def __getitem__(self, index):
        return self.__data_deque[index]

    def clear(self):
        """ 清空数据集 """
        self.__data_deque.clear()


device = torch.device('cuda:0')

# 创建数网络
policy_value_net = PolicyValueNet(board_len=7, n_feature_planes=5, policy_output_dim=168, is_use_gpu=True)
# policy_value_net: PolicyValueNet = torch.load("./model/policy_value_net_100.pth")

# 创建优化器和损失函数
optimizer = Adam(policy_value_net.parameters(), lr=1e-2, weight_decay=1e-4)
criterion = PolicyValueLoss()

# self.lr_scheduler = MultiStepLR(self.optimizer, [1500, 2500], gamma=0.1)
# lr_scheduler = ExponentialLR(optimizer, gamma=0.998)  # 0.998 ** 1000 = 0.135

loaded_data = np.load("../data/processed_data.npz")

planes, best_moves, evals = loaded_data['planes'], loaded_data['best_moves'], loaded_data['evals']

planes, best_moves, evals = planes[:1000000], best_moves[:1000000], evals[:1000000] / 100.0  # 将评估值缩放到 [-1, 1] 范围

planes, best_moves, evals = torch.tensor(planes, dtype=torch.uint8), torch.tensor(best_moves, dtype=torch.int64), torch.tensor(evals, dtype=torch.float32)
best_moves = torch.nn.functional.one_hot(best_moves).to(dtype=torch.uint8)
data_list = []

print("loading data...")
print("data shape:", planes.shape, best_moves.shape, evals.shape)
for i in range(len(best_moves)):
    f, p, zi = planes[i], best_moves[i], evals[i]
    f, p, zi = f.to(device).float(), p.to(device).float(), zi.to(device).float()
    data_list.append((f, p, zi))


dataset = GameDataset(data_list)
print(len(dataset))
data_loader = DataLoader(dataset, batch_size=1000)

policy_value_net.train()
loss_history = []

epoch_num = 500
save_freq = 100

for epoch in range(epoch_num):
    p_bar = tqdm(enumerate(data_loader, 0), ncols=80, total=len(data_loader), desc=f"Epoch {epoch + 1}")
    for i, data in p_bar:
        feature_planes, pi, z = data

        # 前馈
        p_hat, value = policy_value_net(feature_planes)
        # 梯度清零
        optimizer.zero_grad()
        # 计算损失
        loss = criterion(p_hat, pi, value.flatten(), z)
        # 误差反向传播
        loss.backward()
        # 更新参数
        optimizer.step()
        # 学习率退火
        # lr_scheduler.step()

    print(f"Epoch {epoch + 1} Loss: {loss.item():.4f}")
    loss_history.append(loss.item())
    if (epoch + 1) % save_freq == 0:
        torch.save(policy_value_net, f"./model/policy_value_net_{epoch + 1}.pth")
        print(f"Save model to ./model/policy_value_net_{epoch + 1}.pth")

from matplotlib import pyplot as plt

plt.plot(loss_history)
plt.savefig("./loss_history.png")

torch.save(policy_value_net, f"./model/policy_value_net.pth")