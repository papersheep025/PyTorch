# @FileName  :google.py
# @Time      :2023/7/19 11:36
# @Author    :Papersheep


if __name__ == "__main__":
    run_code = 0

import torch
from torch import nn
from d2l import torch as d2l


def corr2d(X, K):
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1,X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i,j] = (X[i:i+h, j:j+w] * K).sum()
    return Y


class Conv2D(nn.Module):
    def __int__(self, kernel_size):
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, X):
        return corr2d(X, self.weight) + self.bias

X = torch.ones((6,8))
X[:,2:6] = 0  # 把中间四列设为0
print(X)

K = torch.tensor([[1.0,-2.0]])  # 设置kernel
Y = corr2d(X,K)
print(Y)
print(corr2d(X.t(), K))
#%% md
#学习由X生成Y的卷积核
