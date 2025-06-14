from matplotlib import pyplot as plt
from collections import deque

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np

import sys,os
sys.path.append(os.getcwd())

from alphazero import PolicyValueNet, PolicyValueLoss


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
policy_value_net = PolicyValueNet(
    board_len=7, n_feature_planes=5, policy_output_dim=168, is_use_gpu=True)
# policy_value_net: PolicyValueNet = torch.load("./model/policy_value_net_100.pth")

# 创建优化器和损失函数
optimizer = Adam(policy_value_net.parameters(), lr=3e-3, weight_decay=0)
criterion = PolicyValueLoss()

# self.lr_scheduler = MultiStepLR(self.optimizer, [1500, 2500], gamma=0.1)
# lr_scheduler = ExponentialLR(optimizer, gamma=0.998)  # 0.998 ** 1000 = 0.135

loaded_data = np.load("./data/processed_data.npz")

planes, best_moves, evals = loaded_data['planes'], loaded_data['best_moves'], loaded_data['evals']

# shuffle data
indices = np.random.permutation(len(planes))
planes, best_moves, evals = planes[indices], best_moves[indices], evals[indices]

# 将评估值缩放到 [-1, 1] 范围
planes, best_moves, evals = planes[:100000], best_moves[:100000], evals[:100000] / 100.0

planes, best_moves, evals = torch.tensor(planes, dtype=torch.uint8), torch.tensor(
    best_moves, dtype=torch.int64), torch.tensor(evals, dtype=torch.float32)
best_moves = torch.nn.functional.one_hot(best_moves).to(dtype=torch.uint8)

train_data_list = []
test_data_list = []

print("loading data...")
print("data shape:", planes.shape, best_moves.shape, evals.shape)
for i in range(len(best_moves)):
    f, pi, z = planes[i], best_moves[i], evals[i]
    f, pi, z = f.to(device).float(), pi.to(
        device).float(), z.to(device).float()

    if i < 10000:
        test_data_list.append((f, pi, z))
    else:
        train_data_list.append((f, pi, z))


dataset = GameDataset(train_data_list)
print(len(dataset))
data_loader = DataLoader(dataset, batch_size=1000)

policy_value_net.train()
loss_history = []

epoch_num = 500
save_freq = 50

for epoch in range(epoch_num):
    p_bar = tqdm(enumerate(data_loader, 0), ncols=80,
                 total=len(data_loader), desc=f"Epoch {epoch + 1}")

    if (epoch + 1) % 10 == 0:
        # Evaluate on test dataset
        test_dataset = GameDataset(test_data_list)
        test_loader = DataLoader(test_dataset, batch_size=1000)
        policy_value_net.eval()
        total_loss = 0.0
        with torch.no_grad():
            for data in test_loader:
                feature_planes, pi, z = data
                p_hat, value = policy_value_net(feature_planes)
                loss = criterion(p_hat, pi, value.flatten(), z, verbose=True)
                total_loss += loss.item() * feature_planes.size(0)
            avg_loss = total_loss / len(test_dataset)
            print(f"Epoch {epoch + 1}, Test Loss: {avg_loss:.4f}")
        policy_value_net.train()

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
        p_bar.set_postfix(loss=loss.item())

    loss_history.append(loss.item())
    if (epoch + 1) % save_freq == 0:
        torch.save(policy_value_net,
                   f"./model/policy_value_net_{epoch + 1}.pth")
        print(f"Save model to ./model/policy_value_net_{epoch + 1}.pth")


plt.plot(loss_history)
plt.savefig("./loss_history.png")

torch.save(policy_value_net, f"./model/policy_value_net.pth")
