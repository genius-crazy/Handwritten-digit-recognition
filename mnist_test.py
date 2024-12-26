import cv2
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data as datautils
from CNN import CNN

# 加载MNIST测试数据集
test_data = datasets.MNIST(
    root='mnist',
    train=False,
    download=True,
    transform=transforms.ToTensor())


# 创建数据加载器
test_loader = datautils.DataLoader(
    dataset=test_data,
    batch_size=64,
    shuffle=True)

# 加载训练好的模型
cnn = torch.load('model\mnist_model.pkl')
cnn.cuda()

# 定义损失函数
loss_func = torch.nn.CrossEntropyLoss()

# 初始化损失和正确预测数量
loss_test = 0
rightValue = 0

# 遍历测试数据集
for index1,(images,labels) in enumerate(test_loader):
    # 将数据移动到GPU
    images = images.cuda()
    labels = labels.cuda()
    # 使用模型进行预测
    outputs = cnn(images)
    _,pred = outputs.max(1)
    # 计算损失
    loss_test += loss_func(outputs,labels)
    # 计算正确预测数量
    rightValue += (pred==labels).sum().item()

    # 将数据移回CPU
    images = images.cpu().numpy()
    labels = labels.cpu().numpy()
    pred = pred.cpu().numpy()
    # 遍历每个样本
    for idx in range(images.shape[0]):
        # 获取图像数据
        im_data = images[idx]
        # 转置图像数据
        im_data = im_data.transpose(1,2,0)
        # 获取真实标签
        im_label = labels[idx]
        # 获取预测标签
        im_pred = pred[idx]
        # 打印预测结果和真实结果
        print(f'预测结果为{im_pred}')
        print(f'真实结果为{im_label}')
        # 显示图像
        cv2.imshow('current_image',im_data)
        # 等待按键
        cv2.waitKey(0)

# 打印损失和准确率
print(f'损失函数为{loss_test},准确率为{rightValue/len(test_data)}')


