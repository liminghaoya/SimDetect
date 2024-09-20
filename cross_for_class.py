import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn
from torchvision import transforms
from PIL import Image
from utils import vgg, tools

# 指定使用 GPU 卡 2
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# 创建输出目录
output_dir = "attention_map"
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)  # 确保创建目录

## 计算 attention map
def attention_map(fm, p=2.5, eps=1e-6):
    am = torch.pow(torch.abs(fm), p)
    am = torch.sum(am, dim=1, keepdim=True)
    norm = torch.norm(am, dim=(2, 3), keepdim=True)
    am = torch.div(am, norm + eps)
    am = nn.functional.interpolate(am, size=(32, 32), mode='bilinear', align_corners=False)
    return am

# 加载模型并初始化
module = vgg.vgg16_bn()
module.load_state_dict(torch.load(
    "/root/root/lmh/backdoor-toolbox_ori/poisoned_train_set/cifar10/badnet_0.000_poison_seed=0/full_base_aug_seed=2333.pt"))

module = nn.DataParallel(module).cuda()
module.eval()

## 数据集加载以及预处理
data_transform_aug = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
])

poisoned_data_path = "/root/root/lmh/backdoor-toolbox_SIBP/poisoned_train_set/cifar10/none_0.000_poison_seed=0/imgs"
poisoned_label_path = "/root/root/lmh/backdoor-toolbox_SIBP/poisoned_train_set/cifar10/none_0.000_poison_seed=0/labels"
poisoned_set = tools.IMG_Dataset(data_dir=poisoned_data_path, label_path=poisoned_label_path,
                                 transforms=data_transform_aug)
poisoned_set_loader = torch.utils.data.DataLoader(poisoned_set, batch_size=256, shuffle=False, **{'pin_memory': True})

# 初始化每个类别的累加器和计数器
num_classes = 10  # CIFAR-10 有 10 个类别
class_attention_sums = {i: [None, None, None, None] for i in range(num_classes)}  # 每个类别的 4 个层的累加器
class_counts = {i: 0 for i in range(num_classes)}  # 每个类别的计数器
class_attention_distributions = {i: 0 for i in range(num_classes)}  # 每个类别的注意力分布数据

## 遍历每个类别，生成交叉注意力图并计算分布数据
for data, target in poisoned_set_loader:
    for idx in range(len(data)):
        img = data[idx].unsqueeze(0).cuda()

        # Forward pass, 获取激活层特征图
        output, activation1, activation2, activation3, activation4 = module.forward(img, return_activation=True)

        # 获取输出类别的 softmax
        output_prob = nn.functional.softmax(output, dim=1)
        predicted_class = torch.argmax(output_prob, dim=1).item()

        # 计算每层的注意力图
        am_list = [
            attention_map(activation1),
            attention_map(activation2),
            attention_map(activation3),
            attention_map(activation4)
        ]

        # 对于每个类别的 4 个层，累加注意力图
        for i, am in enumerate(am_list):
            if class_attention_sums[predicted_class][i] is None:
                class_attention_sums[predicted_class][i] = am
            else:
                class_attention_sums[predicted_class][i] += am

        class_counts[predicted_class] += 1  # 增加该类别的图像计数

# 计算每个类别的平均注意力分布
for class_id in range(num_classes):
    if class_counts[class_id] > 0:  # 确保类别中有图像
        attention_sum = 0
        for i in range(4):  # 4 个层的注意力图累加
            avg_attention = class_attention_sums[class_id][i] / class_counts[class_id]
            attention_sum += avg_attention.sum().item()  # 累加注意力值

        class_attention_distributions[class_id] = attention_sum / 4  # 取 4 个层的平均值

# 绘制每个类别的交叉注意力分布柱状图
plt.bar(class_attention_distributions.keys(), class_attention_distributions.values())
plt.xlabel('class')
plt.ylabel('fenbu')
plt.title('xxxxx')

# 确保目录存在并保存图片
plt.savefig(os.path.join(output_dir, "clean_model_class_attention_distribution.png"))
