from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch import nn
import torch.nn.functional as F
import torch
import os

PROJECT_PATH = "C:/Users/bbben/Desktop/torch"

# 训练数据集
DATA_TRAIN = PROJECT_PATH + "/imgresized/"
# 验证数据集
DATA_TEST = PROJECT_PATH + "/imgtest/"
# 模型保存地址
DATA_MODEL = PROJECT_PATH + "/model/1.model"


def get_transform():
    return transforms.Compose([
            transforms.ToTensor(),
            # 对每个像素点进行归一化
            transforms.Normalize(mean=[0.4, 0.4, 0.4],
                                 std=[0.2, 0.2, 0.2])
        ])

def get_dataset(batch_size=10, num_workers=1):
    data_transform = get_transform()
    # load训练集图片
    train_dataset = ImageFolder(root=DATA_TRAIN, transform=data_transform)
    # load验证集图片
    test_dataset = ImageFolder(root=DATA_TEST, transform=data_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return train_loader, test_loader

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 第一层卷积->激活->池化->Dropout
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.2)
        )
        # 第二层卷积->激活->池化->Dropout
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2)
        )
        # 全连接层 
        self.out = nn.Linear(32 * 8 * 8, 2)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.out(x)
        # 对结果进行log + softmax并输出
        return F.log_softmax(x, dim=1)

# 检查是否有GPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model():
    train_loader, test_loader = get_dataset()
    net = Net().to(DEVICE)
    # 使用Adam优化器
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    for epoch in range(100): #enpochs
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            output = net(x)
            # 使用最大似然 / log似然代价函数
            loss = F.nll_loss(output, y)
            # Pytorch会梯度累计所以需要梯度清零
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 使用Adam进行梯度更新
            optimizer.step()

            if (step + 1) % 3 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch + 1, step * len(x), len(train_loader.dataset),
                    100. * step / len(train_loader), loss.item()))
    # 使用验证集查看模型效果
    test(net, test_loader)
    # 保存模型权重到 config.DATA_MODEL目录
    torch.save(net.state_dict(), DATA_MODEL)
    return net

def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            output = model(x)
            test_loss += F.nll_loss(output, y, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(y.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\ntest loss={:.4f}, accuracy={:.4f}\n'.format(test_loss, float(correct) / len(test_loader.dataset)))

def predict_model(image):
    data_transform = get_transform()
    # 对图片进行预处理，同训练的时候一样
    image = data_transform(image)
    image = image.view(-1, 3, 32, 32)
    net = Net().to(DEVICE)
    # 加载模型参数权重
    net.load_state_dict(torch.load(DATA_MODEL))
    output = net(image.to(DEVICE))
    # 输出概率
    return -output[0][1].item()

def train():
    print("train start")
    train_model()
    print("train done.")

def main():
    print(__name__)
    train()

if __name__ == '__main__':
    main()