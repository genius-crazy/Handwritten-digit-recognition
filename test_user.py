import os
import torch
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# 加载预训练模型
cnn = torch.load('model/mnist_model.pkl')
cnn = cnn.cuda()  

# 定义数据预处理（转换为Tensor并标准化）
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # 确保是灰度图
    transforms.Resize((28, 28)),  # 调整大小为28x28
    transforms.ToTensor(),  # 转换为Tensor
    transforms.Normalize((0.5,), (0.5,))  # 使用与训练时相同的标准化
])

# 设置路径
folder_path = 'user_testing'  
labels_file = 'user_testing\labels.ini'  

# 加载标签文件
labels_dict = {}
with open(labels_file, 'r') as f:
    for line in f:
        parts = line.strip().split()
        filename = parts[0]
        label = int(parts[1])
        labels_dict[filename] = label

# 初始化变量用于计算准确率
correct_predictions = 0
total_predictions = 0

flag = False
if input("是否需要显示图片？(y/n): ") == "y":
    flag = True

# 遍历文件夹中的所有图片
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)

    # 处理图片文件
    if filename.endswith('.png') or filename.endswith('.jpg'):
        try:
            # 加载并预处理图片
            image = Image.open(file_path)
            image_for_display = image.copy()  # 复制图片以供显示
            image = transform(image).unsqueeze(0)  # 添加批次维度 (batch_size=1)
            image = image.cuda()  

            # 模型预测
            cnn.eval()  # 设置为评估模式
            with torch.no_grad():  # 不计算梯度
                output = cnn(image)  # 进行前向传播
                _, predicted = output.max(1)  # 获取预测的值
            
            # 获取真实标签
            true_label = labels_dict.get(filename, None)
            if true_label is not None:
                # 计算准确率
                if predicted.item() == true_label:
                    correct_predictions += 1
                total_predictions += 1


            # 输出结果
            print(f"图片 {filename} 的预测结果是: {predicted.item()}")

            if flag:
                # 使用matplotlib显示图片并标注预测结果
                plt.imshow(image_for_display, cmap='gray')  # 使用原图
                plt.title(f"Predicted: {predicted.item()}")  # 显示预测结果
                plt.axis('off')  # 关闭坐标轴
                plt.show()  # 显示图片
        except Exception as e:
            print(f"无法处理图片 {filename}: {e}")

print(f"总预测数: {total_predictions}")
print(f"准确率: {correct_predictions / total_predictions}")