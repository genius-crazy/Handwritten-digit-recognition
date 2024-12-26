# 导入torch库
import torch
# 导入torchvision库中的datasets模块
import torchvision.datasets as datasets
# 导入torchvision库中的transforms模块
import torchvision.transforms as transforms
# 导入torch.utils.data模块
import torch.utils.data as datautils
# 导入CNN模型
from CNN import CNN

# 加载训练集数据
train_data = datasets.MNIST(
    root='mnist',  # 数据集的根目录
    train=True,  # 是否为训练集
    download=True,  # 是否下载数据集
    transform=transforms.ToTensor())  # 数据预处理

# 加载测试集数据
test_data = datasets.MNIST(
    root='mnist',  # 数据集的根目录
    train=False,  # 是否为训练集
    download=True,  # 是否下载数据集
    transform=transforms.ToTensor())  # 数据预处理


# 创建训练集数据加载器
train_loader = datautils.DataLoader(
    dataset=train_data,  # 数据集
    batch_size=64,  # 每个批次的数据量
    shuffle=True)  # 是否打乱数据

# 创建测试集数据加载器
test_loader = datautils.DataLoader(
    dataset=test_data,  # 数据集
    batch_size=64,  # 每个批次的数据量
    shuffle=True)  # 是否打乱数据



# 创建CNN模型
cnn = CNN()
# 将模型移动到GPU上
cnn = cnn.cuda()


# 定义损失函数
loss_func = torch.nn.CrossEntropyLoss()
# 定义优化器
optimizer = torch.optim.Adam(cnn.parameters(),lr=0.01)

# 训练模型
for epoch in range(10): 
    # 遍历训练集
    for index1,(images,labels) in enumerate(train_loader):
        # 将数据移动到GPU上
        images = images.cuda()
        labels = labels.cuda()
        # 前向传播
        outputs = cnn(images)
        # 计算损失
        loss = loss_func(outputs,labels)
        # 梯度清零
        optimizer.zero_grad()
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()
        # 打印损失
        print(f'第{epoch+1}轮训练,第{index1+1}/{len(train_data)//64+1}批次训练，损失函数为{loss.item()}')

    # 测试模型
    loss_test = 0
    rightValue = 0
    for index2,(images,labels) in enumerate(test_loader):
        # 将数据移动到GPU上
        images = images.cuda()
        labels = labels.cuda()
        # 前向传播
        outputs = cnn(images)
        # 计算损失
        loss_test += loss_func(outputs,labels)
        # 获取预测结果
        _,pred = outputs.max(1)
        # 计算准确率
        rightValue += (pred==labels).sum().item()
        # 打印损失和准确率
        print(f'第{epoch+1}轮测试集验证,第{index2+1}/{len(test_loader)}批次，损失函数为{loss_test},准确率为{rightValue/len(test_data)}')
    
# 保存模型
torch.save(cnn,'model/mnist_model.pkl')
        
    















