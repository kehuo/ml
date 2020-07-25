# @File: pytorch_linear_classifier
# @Author: Kevin Huo
# @LastUpdate: 7/20/2020 7:37 PM

import torch
import numpy as np
import matplotlib as mat
import matplotlib.pyplot as plt

# pytorch 的 hello world, 构造一个最简单的线性分类模型.

# 1 给出一些点
x_train_nparray = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                   [9.779], [6.182], [7.59], [2.167], [7.042],
                   [10.791], [5.313], [7.997], [3.1]],
                   dtype=np.float32)

y_train_nparray = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
                   [3.366], [2.596], [2.53], [1.221], [2.827],
                   [3.465], [1.65], [2.904], [1.3]],
                   dtype=np.float32)

# 2 画出散点图需要用 scatter
# plt.scatter(x_train, y_train)
# plt.show()

# 第二部分 开始pytorch 模型处理
# 2.1 将 np.array 转换成 torch 里面的张量 Tensor:
x_train = torch.from_numpy(x_train_nparray)
y_train = torch.from_numpy(y_train_nparray)


# 2.2 建立线性回归模型
class LinearRegressionModel(torch.nn.Module):
    def __init__(self):
        # 这一行super的代码，是必须要的. 固定套路
        super(LinearRegressionModel, self).__init__()
        # 输入和输出都是 1 维, 即 y = wx + b
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        output = self.linear(x)
        return output


# 2.3 如果支持GPU则用GPU，否则用CPU
if torch.cuda.is_available():
    my_model = LinearRegressionModel().cuda()
else:
    my_model = LinearRegressionModel()


# 2.4 定义损失函数 (均方误差) 和优化函数(梯度下降)
criterion = torch.nn.MSELoss()
# lr是学习率
optimizer = torch.optim.SGD(my_model.parameters(), lr=0.003)


# 第三部分 - 训练模型
# todo 3.1 训练
def main():
    epoches = 100
    for epoch in range(epoches):
        if torch.cuda.is_available():
            inputs = torch.autograd.Variable(x_train).cuda()
            target = torch.autograd.Variable(y_train).cuda()
        else:
            inputs = torch.autograd.Variable(x_train)
            target = torch.autograd.Variable(y_train)

        # forward
        # todo - 书里这是 my_model(inputs), 但是我觉得应该是 my_model.forward(inputs), 之后再研究谁对谁错.
        fwd_out = my_model.forward(inputs)
        loss = criterion(fwd_out, target)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print some info
        if (epoch + 1) % 20 == 0:
            print("Epoch {}/{}, Loss: {}".format(epoch + 1, epoches, loss.data))


if __name__ == '__main__':
    main()
