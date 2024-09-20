import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn
from torchvision import transforms
from utils import vgg, tools

# 指定使用 GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

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

# 计算 Frobenius 范数
def frobenius_norm_diff(am1, am2):
    diff = am1 - am2
    return torch.norm(diff, p='fro').item()

# 加载模型并初始化
module = vgg.vgg16_bn(num_classes=43)
module.load_state_dict(torch.load(
    "/root/root/lmh/backdoor-toolbox_ori/poisoned_train_set/gtsrb/none_0.000_poison_seed=0/full_base_aug_seed=2333.pt"))

module = nn.DataParallel(module).cuda()
module.eval()

## 数据集加载以及预处理
data_transform_aug = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
])

clean_data_path = "/root/root/lmh/backdoor-toolbox_SIBP/poisoned_train_set/gtsrb/none_0.000_poison_seed=0/imgs"
clean_label_path = "/root/root/lmh/backdoor-toolbox_SIBP/poisoned_train_set/gtsrb/none_0.000_poison_seed=0/labels"
dataset_set = tools.IMG_Dataset(data_dir=clean_data_path, label_path=clean_label_path,
                                 transforms=data_transform_aug)
dataset_set_loader = torch.utils.data.DataLoader(dataset_set, batch_size=256, shuffle=False, **{'pin_memory': True})

# 初始化每个类别的累加器和计数器
num_classes = 43  # CIFAR-10 有 10 个类别   GTSRB有43个类别
class_attention_sums = {i: [None, None, None, None] for i in range(num_classes)}  # 每个类别的 4 个层的累加器
class_counts = {i: 0 for i in range(num_classes)}  # 每个类别的计数器
class_attention_distributions = {i: 0 for i in range(num_classes)}  # 每个类别的注意力分布数据

## 遍历每个类别，生成交叉注意力图并计算分布数据
for data, target in dataset_set_loader:
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

# 计算每层的 Frobenius 范数差异并进行加权平均
frobenius_matrix = np.zeros((num_classes, num_classes, 4))

for layer in range(4):
    for i in range(num_classes):
        if class_counts[i] == 0:
            continue
        for j in range(i + 1, num_classes):
            if class_counts[j] == 0:
                continue
            avg_attention_i = class_attention_sums[i][layer] / class_counts[i]
            avg_attention_j = class_attention_sums[j][layer] / class_counts[j]
            frobenius_matrix[i, j, layer] = frobenius_norm_diff(avg_attention_i, avg_attention_j)
            frobenius_matrix[j, i, layer] = frobenius_matrix[i, j, layer]

# 计算每个类别的标准差
std_diff_per_class = np.std(frobenius_matrix, axis=2)  # 计算每个类别对的标准差

# 可视化各类标准差矩阵
plt.imshow(std_diff_per_class, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.title('Class Attention Frobenius Norm Standard Deviation Matrix')
plt.savefig(os.path.join(output_dir, "att_class_attention_frobenius_norm_std_matrix.png"))
plt.show()


# 计算同化现象的分数
def calculate_assimilation_score(std_diff_per_class, poison_class=0):
    non_poison_classes = [i for i in range(std_diff_per_class.shape[0]) if i != poison_class]
    total_std_diff = 0
    count = 0
    for i in non_poison_classes:
        for j in non_poison_classes:
            if i != j:
                total_std_diff += std_diff_per_class[i, j]
                count += 1
    return total_std_diff / count if count > 0 else 0


for i in range(num_classes):
    assimilation_score = calculate_assimilation_score(std_diff_per_class, poison_class=i)
    print(f"Assimilation Score: {assimilation_score}")
