import torch

# 定义一个CNN类，继承自torch.nn.Module
class CNN(torch.nn.Module):
    # 初始化函数
    def __init__(self):
        super(CNN, self).__init__()
        # 定义卷积层
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1,out_channels=32,kernel_size=5,padding=2),  # 输入通道数为1，输出通道数为32，卷积核大小为5，padding为2

            torch.nn.BatchNorm2d(32), # 对32个通道进行批量归一化

            torch.nn.ReLU(), # 激活函数

            torch.nn.MaxPool2d(kernel_size=2), #14*14
        )
        self.fc = torch.nn.Linear(in_features=32*14*14,out_features=10)

    def forward(self,x):
        out = self.conv(x)
        out = out.view(out.size(0),-1)
        out = self.fc(out)
        return out




