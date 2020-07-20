# @File: pytorch_mnist.py
# @Author: Kevin Huo
# @LastUpdate: 7/19/2020 4:16 PM


import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


# 第一部分 - 定义全局变量
BATCH_SIZE = 512
EPOCHS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 第二部分 - 下载训练集
# 2.1 定义 transform
my_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1037,), (0.3081,))
])

# 2.2 - 下载
train_datasets = datasets.MNIST(root='data', train=True, download=True, transform=my_transform)
test_datasets = datasets.MNIST(root='data', train=False, transform=my_transform)

# 2.3 - 加载到迭代器 loader 中
my_train_loader = DataLoader(train_datasets, batch_size=BATCH_SIZE, shuffle=True)
my_test_loader = DataLoader(test_datasets, batch_size=BATCH_SIZE, shuffle=True)


###############################
# 第三部分 - 定义神经网络的模型
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 1*1*28*28
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.conv2 = nn.Conv2d(10, 20, 3)
        self.fc1 = nn.Linear(20 * 10 * 10, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        in_size = x.size(0)
        out = self.conv1(x)  # 1* 10 * 24 *24
        out = F.relu(out)
        out = F.max_pool2d(out, 2, 2)  # 1* 10 * 12 * 12
        out = self.conv2(out)  # 1* 20 * 10 * 10
        out = F.relu(out)
        out = out.view(in_size, -1)  # 1 * 2000
        out = self.fc1(out)  # 1 * 500
        out = F.relu(out)
        out = self.fc2(out)  # 1 * 10
        out = F.log_softmax(out, dim=1)
        return out


################################
# 第四部分 - 生成模型和优化器
# 这里用到的不是最基础的 梯度下降优化器，而是高级的 Adam 优化器.
my_model = MyModel().to(DEVICE)
my_optimizer = optim.Adam(my_model.parameters())


##############################
# 第五部分 - 训练和测试
# 5.1 定义训练函数
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if (batch_idx + 1) % 30 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data),
                len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item()
            ))


# 5.2 定义测试函数
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum')  # 将这一批的损失相加
            pred = output.max(1, keepdim=True)[1]  # 找到概率最大的下标
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print("\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%) \n".format(
        test_loss,
        correct,
        len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)
    ))


########################################
# 第六部分 - 开始训练和测试
def main():
    for epoch in range(1, EPOCHS + 1):
        train(my_model,  DEVICE, my_train_loader, my_optimizer, epoch)
        test(my_model, DEVICE, my_test_loader)


# 第七部分 - 保存模型
# 其实有2中保存方式，其一是只保存模型的参数，其二是将模型整个保存下来。
# 我选择方法2，因为导入或者再次使用时，比较方便。

torch.save(my_model, ".\\saved_models\\mnist.pth")
# 如果要加载，用以下代码 (加载可能需要点时间)
mnist_loaded_model = torch.load(".\\saved_models\\mnist.pth")

if __name__ == '__main__':
    main()
