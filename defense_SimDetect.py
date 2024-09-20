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
output_dir = "sample_cda_output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

# 计算 attention map
def attention_map(fm, p=1.5, eps=1e-6):
    am = torch.pow(torch.abs(fm), p)
    am = torch.sum(am, dim=1, keepdim=True)
    norm = torch.norm(am, dim=(2, 3), keepdim=True)
    am = torch.div(am, norm + eps)
    am = nn.functional.interpolate(am, size=(32, 32), mode='bilinear', align_corners=False)
    return am

# 计算协方差矩阵
def covariance_matrix(am):
    am_flat = am.view(am.size(0), -1)  # 展平注意力图
    cov_matrix = torch.cov(am_flat)    # 计算协方差矩阵
    return cov_matrix




import time

# 记录开始时间
start_time = time.time()

# 加载模型并初始化
module = vgg.vgg16_bn(num_classes=10)
module.load_state_dict(torch.load(
    "/root/root/lmh/backdoor-toolbox_ori/poisoned_train_set/cifar10/none_0.000_poison_seed=0/full_base_aug_seed=2333.pt"))
module = nn.DataParallel(module).cuda()
module.eval()

## 数据集加载以及预处理
data_transform_aug = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
])

clean_data_path = "/root/root/lmh/backdoor-toolbox_SIBP/poisoned_train_set/cifar10/none_0.000_poison_seed=0/imgs"
clean_label_path = "/root/root/lmh/backdoor-toolbox_SIBP/poisoned_train_set/cifar10/none_0.000_poison_seed=0/labels"
dataset_set = tools.IMG_Dataset(data_dir=clean_data_path, label_path=clean_label_path,
                                 transforms=data_transform_aug)
dataset_set_loader = torch.utils.data.DataLoader(dataset_set, batch_size=256, shuffle=False, **{'pin_memory': True})

# 初始化每个类别的协方差矩阵累加器和计数器
num_classes = 10  # CIFAR-10 有 10 个类别
class_covariance_sums = {i: [None, None, None, None] for i in range(num_classes)}  # 每个类别的 4 个层的累加器
class_counts = {i: 0 for i in range(num_classes)}  # 每个类别的计数器

# 用于存储每个样本的协方差差异
sample_cda_scores = []

poison_indices=[]
# 你的索引列表
indexes = [28063, 41664, 5888, 29518, 33249, 13086, 47685, 19961, 35782, 14949, 33451, 48266, 9413, 47332, 14169, 1708, 44743, 11849, 8382, 45666, 23899, 5323, 45163, 42190, 46992, 24365, 11237, 29879, 48986, 7898, 34664, 35927, 8425, 18614, 20039, 4861, 10675, 8515, 44464, 11255, 25880, 48580, 30914, 9096, 46200, 35798, 46922, 12993, 7093, 6425, 21924, 22164, 35820, 32934, 33360, 36610, 9057, 5702, 34693, 18009, 17458, 37177, 29837, 6190, 27873, 1466, 49510, 40648, 33489, 33214, 24881, 12991, 35209, 7612, 27623, 23863, 26824, 42103, 19011, 29367, 5924, 33584, 20355, 46756, 15396, 13206, 15483, 13833, 15580, 11014, 28300, 3585, 15923, 14592, 44498, 18012, 20383, 34109, 10990, 39286, 36337, 31004, 47320, 7842, 23221, 42878, 3478, 26488, 28627, 19196, 41121, 12252, 35544, 4940, 2277, 25457, 27438, 4537, 9755, 33186, 36312, 911, 37439, 33542, 38519, 29544, 5757, 36100, 26430, 20190, 49530, 25644, 39409, 37078, 30517, 14079, 1335, 33670, 44645, 468, 32267, 13106, 26815, 11783, 41855, 25789, 43564, 41821, 23155, 28360, 19632, 7095, 13473, 25290, 6370, 2453, 31674, 34121, 27026, 41868, 25350, 3940, 40945, 33130, 14350, 13594, 38198, 44866, 19084, 9438, 4516, 43727, 18315, 10205, 38285, 43998, 45656, 35195, 42722, 47693, 349, 39877, 32378, 44116, 37493, 3570, 11979, 21761, 34434, 35546, 1434, 43160, 6858, 43947, 22449, 29884, 22967, 12130, 5988, 29754, 43750, 5210, 5883, 36927, 22159, 46576, 3024, 373, 25385, 33347, 39959, 4577, 43667, 26723, 2171, 21158, 20722, 4027, 2429, 38716, 24791, 16182, 40619, 37949, 3644, 27425, 36530, 33117, 37738, 32002, 24445, 12052, 18925, 22665, 33000, 44653, 48489, 41574, 37063, 2459, 22258, 557, 28216, 13326, 4601, 19818, 31164, 25352, 29257, 9799, 16065, 10798, 43454, 42867, 15113, 39080, 6054, 49339, 22993, 43944, 47726, 48327, 1594, 4476, 12265, 35524, 29748, 3647, 23171, 15452, 44245, 22506, 11615, 37978, 13489, 16186, 26200, 39201, 39819, 41212, 36252, 26667, 4759, 10266, 23746, 23836, 39716, 34724, 6867, 33923, 44806, 34745, 3962, 25362, 30257, 36155, 49164, 23449, 27710, 47676, 19106, 19664, 48792, 4061, 47925, 46631, 18839, 1507, 33933, 39026, 8813, 8023, 39006, 34748, 45357, 13880, 29634, 48024, 17100, 15436, 25066, 11251, 23825, 42945, 16420, 19216, 28978, 20328, 18900, 48174, 41243, 34676, 21533, 47834, 5331, 48520, 35731, 42618, 18979, 5404, 29364, 17866, 39392, 4509, 18906, 29653, 10024, 22249, 8132, 46802, 36068, 20743, 33654, 30141, 25668, 12363, 7599, 10667, 23984, 18505, 16115, 39723, 812, 21522, 26040, 8083, 20606, 34012, 15541, 49100, 22276, 40287, 8953, 46367, 28512, 43442, 16625, 29783, 20759, 48428, 36981, 31693, 23157, 2006, 12915, 31083, 28919, 26913, 6919, 10297, 28179, 36119, 26162, 35179, 38566, 1540, 40432, 47711, 39121, 46655, 1266, 39447, 6142, 46192, 11022, 1913, 7174, 24676, 42201, 24931, 25624, 44037, 7168, 44454, 21095, 12553, 11475, 25337, 3049, 26744, 24681, 46603, 25391, 37646, 4249, 37240, 17174, 35873, 32149, 14334, 21782, 7060, 34131, 19286, 22189, 42439, 757, 44165, 17353, 29666, 47336, 43525, 7979, 199, 4571, 2107, 27202, 27007, 42435, 29190, 38321, 35952, 2473, 28400, 3102, 48492, 28311, 15572, 37872, 14689, 36038, 30539, 18572, 18406, 46366, 28515, 38937, 8209, 39281, 10499, 1432, 22408, 32522, 4240, 40401, 43219, 13800, 3574, 45123, 16230, 20454, 3884, 8072, 1674, 43211, 32251, 9119, 16529, 32761, 20924, 26376, 22692, 4202, 46394, 36386, 34780, 9375, 11732, 2863, 41030, 31757, 14909, 46738, 9430, 30069, 32318, 8160, 24622, 47271, 2490, 19386, 21760, 11572, 25546, 1243, 30971, 37182, 18657, 4490, 39863, 7444, 36231, 6067, 40618, 28068, 18854, 11302, 6013, 9945, 16010, 37522, 7502, 41702, 37909, 8376, 5371, 8912, 46439, 46224, 11023, 16954, 16048, 42852, 26300, 17917, 39012, 28972, 32341, 8833, 7490, 8410, 15304, 36545, 49490, 11958, 35373, 15735, 39104, 28401, 17490, 8934, 1872, 48557, 35369, 29151, 31481, 35599, 49513, 7914, 25242, 42559, 25502, 41055, 38368, 36289, 10855, 4854, 32613, 12499, 38091, 7674, 3063, 45956, 45085, 32683, 5955, 9312, 32931, 48430, 3362, 44136, 26840, 43483, 7187, 26693, 18358, 33421, 31471, 7663, 48516, 7434, 3042, 25027, 41242, 18697, 10008, 22608, 24983, 42543, 38670, 22796, 42283, 47468, 43709, 44243, 15137, 417, 4060, 2145, 4311, 31127, 29860, 16596, 3713, 39927, 11356, 10812, 24514, 27047, 12133, 9859, 598, 49346, 45509, 47431, 46981, 35343, 38222, 28921, 42754, 47581, 24943, 28305, 47849, 3352, 42288, 20473, 34022, 36363, 11018, 46362, 20532, 26571, 26124, 23398, 31970, 46324, 43277, 46964, 42929, 45067, 15900, 43754, 43548, 5566, 40679, 1988, 7907, 32247, 36886, 28185, 28625, 25439, 29244, 16375, 35248, 35286, 48254, 35414, 25220, 27552, 26177, 44176, 29174, 39763, 41355, 1338, 35325, 36180, 2342, 12344, 21195, 4477, 40768, 33662, 32062, 41762, 2345, 32508, 15887, 19370, 41222, 37325, 23781, 33453, 45457, 12477, 33732, 28036, 21869, 42179, 30537, 27444, 26708, 18763, 14275, 27301, 11240, 16952, 36314, 11989, 21975, 44968, 6373, 46750, 10399, 16292, 25820, 45601, 17959, 37561, 27954, 2258, 1012, 8421, 14796, 30233, 47852, 938, 9696, 19730, 48148, 25275, 45004, 31201, 17224, 11514, 7377, 1428, 28742, 4925, 39288, 31086, 902, 37022, 42587, 32971, 23462, 40741, 45387, 19814, 44470, 44592, 20045, 28570, 37566, 19237, 35446, 25696, 38536, 8849, 27075, 18861, 35551, 36356, 16460, 38782, 41124, 3918, 40556, 1195, 30429, 18920, 24713, 37517, 19009, 44946, 44741, 40635, 40504, 36531, 11078, 43475, 44799, 49439, 49633, 26846, 14343, 13507, 43731, 14329, 7347, 25763, 26724, 11509, 13420, 3549, 41152, 33895, 36195, 30521, 4721, 18215, 708, 23848, 14203, 23797, 26049, 43253, 20070, 46831, 45904, 8782, 10847, 17178, 21758, 30710, 44035, 41863, 13357, 28558, 10050, 41467, 8359, 4192, 40113, 13861, 24447, 116, 20205, 39396, 48145, 5873, 9966, 17054, 23924, 1147, 12078, 37103, 33160, 40984, 43965, 44616, 3880, 14856, 11431, 22026, 46701, 32326, 20758, 10686, 11775, 48424, 42019, 11305, 17509, 40150, 20349, 5803, 30324, 4774, 4952, 47446, 8898, 17829, 27981, 36055, 47691, 25187, 5094, 48748, 3136, 30467, 16963, 37788, 11812, 26018, 29596, 30564, 5174, 3721, 15500, 7985, 43458, 22684, 28699, 45304, 7423, 26495, 17133, 30895, 18302, 22254, 46613, 39907, 23096, 8810, 11762, 36427, 12270, 48643, 18955, 20265, 17355, 6112, 10690, 13400, 21166, 22109, 687, 32968, 12297, 2156, 36228, 38208, 332, 7801, 37999, 37843, 9934, 23175, 14138, 15959, 7344, 44731, 11208, 5642, 18386, 31435, 4778, 43996, 22540, 15755, 21573, 12234, 10159, 4991, 10265, 15884, 36148, 7445, 19067, 10776, 10493, 19390, 21723, 293, 37728, 24605, 46513, 7859, 6687, 24587, 16838, 33510, 2034, 10804, 49635, 25599, 35147, 14487, 33242, 14666, 48252, 3903, 2946, 7520, 42628, 2639, 40872, 45445, 26296, 11772, 16939, 40328, 30829, 43757, 38233, 14732, 30487, 32617, 49091, 38112, 21231, 25243, 16866, 21777, 39709, 22973, 17257]


poison_indices=[349, 468, 911, 1335, 1434, 1466, 1708, 2277, 2453, 3478, 3570, 3585, 3940, 4516, 4537, 4861, 4940, 5323, 5702, 5757, 5888, 5924, 5988, 6190, 6370, 6425, 6858, 7093, 7095, 7612, 7842, 7898, 8382, 8425, 8515, 9057, 9096, 9413, 9438, 9755, 10205, 10675, 10990, 11014, 11237, 11255, 11783, 11849, 11979, 12130, 12252, 12991, 12993, 13086, 13106, 13206, 13473, 13594, 13833, 14079, 14169, 14350, 14592, 14949, 15396, 15483, 15580, 15923, 17458, 18009, 18012, 18315, 18614, 19011, 19084, 19196, 19632, 19961, 20039, 20190, 20355, 20383, 21761, 21924, 22164, 22449, 22967, 23155, 23221, 23863, 23899, 24365, 24881, 25290, 25350, 25457, 25644, 25789, 25880, 26430, 26488, 26815, 26824, 27026, 27438, 27623, 27873, 28063, 28300, 28360, 28627, 29367, 29518, 29544, 29754, 29837, 29879, 29884, 30517, 30914, 31004, 31674, 32267, 32378, 32934, 33130, 33186, 33214, 33249, 33360, 33451, 33489, 33542, 33584, 33670, 34109, 34121, 34434, 34664, 34693, 35195, 35209, 35544, 35546, 35782, 35798, 35820, 35927, 36100, 36312, 36337, 36610, 37078, 37177, 37439, 37493, 38198, 38285, 38519, 39286, 39409, 39877, 40648, 40945, 41121, 41664, 41821, 41855, 41868, 42103, 42190, 42722, 42878, 43160, 43564, 43727, 43947, 43998, 44116, 44464, 44498, 44645, 44743, 44866, 45163, 45656, 45666, 46200, 46756, 46922, 46992, 47320, 47332, 47685, 47693, 48266, 48580, 48986, 49510, 49530]





from collections import Counter

# 存储每个类别的频次
label_frequencies = {}

# 遍历每个类别，生成协方差矩阵并计算样本级别的差异
for batch_idx, (data, target) in enumerate(dataset_set_loader):
    for idx in range(len(data)):
        img = data[idx].unsqueeze(0).cuda()
        label = target[idx].item()
        data_index = indexes[batch_idx * len(data) + idx]  # 获取样本对应的索引

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

        # 计算样本的协方差矩阵
        sample_cov_matrices = [covariance_matrix(am) for am in am_list]

        # 如果是正确分类的样本，将其协方差矩阵与当前类别的平均协方差矩阵进行比较
        if predicted_class == label:
            for i, sample_cov_matrix in enumerate(sample_cov_matrices):
                # 更新类别的平均协方差矩阵
                if class_covariance_sums[label][i] is None:
                    class_covariance_sums[label][i] = sample_cov_matrix
                else:
                    class_covariance_sums[label][i] += sample_cov_matrix

            class_counts[label] += 1  # 增加该类别的样本计数

        # 计算样本的 CDA 分数与其所属类别的平均协方差矩阵的差异
        if class_counts[label] > 0:
            cda_scores = []
            for i, sample_cov_matrix in enumerate(sample_cov_matrices):
                avg_cov_matrix = class_covariance_sums[label][i] / class_counts[label]
                cda_score = torch.norm(sample_cov_matrix - avg_cov_matrix, p='fro').item()
                cda_scores.append(cda_score)

            # 存储每个样本的 CDA 分数（针对 4 个激活层的平均值），包括数据索引
            sample_cda_scores.append((data_index, label, predicted_class, np.mean(cda_scores)))

# 将样本 CDA 分数排序，找出异常样本
sample_cda_scores = sorted(sample_cda_scores, key=lambda x: x[3], reverse=True)

# 打印最可疑的中毒样本（CDA 分数最高的样本）
suspect_indices = []
for i in range(10):  # 输出前 30 个最可疑样本
    data_index, label, predicted_class, cda_score = sample_cda_scores[i]
    suspect_indices.append(data_index)

    # 记录每个 label 的频次
    if label not in label_frequencies:
        label_frequencies[label] = 0
    label_frequencies[label] += 1

    print(f"Sample {i}: Index={data_index}, Label={label}, Predicted={predicted_class}, CDA Score={cda_score}")

# 找出频次最高的 Label
sorted_labels = sorted(label_frequencies.items(), key=lambda item: item[1], reverse=True)

# 打印频次最高的怀疑类别
most_suspect_label = sorted_labels[0][0]
print(f"\nMost frequent suspect label: {most_suspect_label}")

# 打印次怀疑类别
if len(sorted_labels) > 1:
    second_suspect_label = sorted_labels[1][0]
    print(f"Second most frequent suspect label: {second_suspect_label}")
else:
    print("There is no second suspect label, only one label was detected.")

# # 比较 suspect_indices 和 poison_indices 计算准确率
# correct_detections = [idx for idx in suspect_indices if idx in poison_indices]
# accuracy = len(correct_detections) / len(suspect_indices) * 100
# print(f"Accuracy of detecting poisoned samples: {accuracy:.2f}%")


# 已找到的最可疑类别
most_suspect_label = sorted_labels[0][0]

# 从检测到的样本中筛选出属于最可疑类别的样本
suspect_samples = [(index, label, predicted_class, cda_score) for index, label, predicted_class, cda_score in sample_cda_scores if label == most_suspect_label]

# 按照 CDA 分数从高到低排序
suspect_samples_sorted = sorted(suspect_samples, key=lambda x: x[3], reverse=True)

# 选择最可疑的前 10 个样本
top_10_suspect_samples = suspect_samples_sorted[:10]

# 打印最可疑的十个样本的索引
print("\nTop 10 suspect sample indices:")
for i, (index, label, predicted_class, cda_score) in enumerate(top_10_suspect_samples):
    print(f"Sample {i+1}: Index={index}, Label={label}, Predicted={predicted_class}, CDA Score={cda_score}")



# # 检查选出的样本中有多少在中毒索引列表中
# correct_predictions = [index for index, _, _, _ in top_10_suspect_samples if index in poison_indices]
#
# # 计算准确率
# accuracy = len(correct_predictions) / len(top_10_suspect_samples) * 100
# print(f"\nAccuracy of top 10 suspect samples: {accuracy:.2f}%")



# # 将样本 CDA 分数排序，找出异常样本
# sample_cda_scores = sorted(sample_cda_scores, key=lambda x: x[4], reverse=True)
#
# # 打印或保存最可疑的中毒样本（CDA 分数最高的样本），带有图像索引
# for i in range(20):  # 输出前 10 个最可疑样本
#     img_index, img, label, predicted_class, cda_score = sample_cda_scores[i]
#     print(f"Sample {i}: Index={img_index}, Label={label}, Predicted={predicted_class}, CDA Score={cda_score}")

# 记录结束时间
end_time = time.time()

# 计算运行时间
execution_time = end_time - start_time
print(f"代码运行时间: {execution_time:.6f} 秒")

# #以下是结合同化程度后的可疑类别分析
# print('以下是结合同化程度后的可疑类别分析')
# # 计算类别的特征均值（针对每一层）
# def calculate_class_mean_for_all_layers(activation_maps_all_layers, labels, num_classes):
#     # 初始化用于存储每层的类别均值
#     class_means = {i: [None for _ in range(4)] for i in range(num_classes)}
#     class_counts = {i: 0 for i in range(num_classes)}
#
#     for idx, (am_list, label) in enumerate(zip(activation_maps_all_layers, labels)):
#         for layer_idx, am in enumerate(am_list):
#             if class_means[label][layer_idx] is None:
#                 class_means[label][layer_idx] = am
#             else:
#                 class_means[label][layer_idx] += am
#         class_counts[label] += 1
#
#     # 计算每个类别每一层的均值
#     for cls in range(num_classes):
#         for layer_idx in range(4):
#             if class_means[cls][layer_idx] is not None:
#                 class_means[cls][layer_idx] /= class_counts[cls]  # 计算每层的均值特征
#
#     return class_means
#
# # 计算样本与其类别每层特征均值的相似度
# def calculate_similarity_with_class_mean_for_all_layers(am_list, class_mean_list):
#     similarity_scores = []
#     for layer_idx in range(4):
#         similarity = nn.functional.cosine_similarity(am_list[layer_idx].view(am_list[layer_idx].size(0), -1),
#                                                      class_mean_list[layer_idx].view(1, -1))
#         similarity_scores.append(similarity.item())
#     return np.mean(similarity_scores)  # 返回四层的相似度的平均值
#
# # 将同化程度与 CDA 分数结合，改进中毒样本识别
# def identify_poisoned_samples_with_all_layers(sample_cda_scores, similarity_scores, threshold=0.9):
#     combined_scores = []
#
#     for idx, (data_index, label, predicted_class, cda_score) in enumerate(sample_cda_scores):
#         similarity_score = similarity_scores[idx]
#         combined_score = cda_score * (1 - similarity_score)  # 根据同化程度调整 CDA 分数
#         combined_scores.append((data_index, label, predicted_class, combined_score))
#
#     # 根据结合后的分数进行排序，找出最可疑的样本
#     combined_scores = sorted(combined_scores, key=lambda x: x[3], reverse=True)
#     return combined_scores
#
# # 在主循环中应用
# activation_map_list_all_layers = []  # 用于存储每个样本的所有激活层的 activation maps
# label_list = []  # 用于存储每个样本的标签
#
# for batch_idx, (data, target) in enumerate(dataset_set_loader):
#     for idx in range(len(data)):
#         img = data[idx].unsqueeze(0).cuda()
#         label = target[idx].item()
#
#         # Forward pass, 获取四层激活层特征图
#         output, activation1, activation2, activation3, activation4 = module.forward(img, return_activation=True)
#
#         # 将四层 activation maps 和 label 存储起来
#         activation_map_list_all_layers.append([activation1, activation2, activation3, activation4])
#         label_list.append(label)
#
# # 计算每个类别的平均特征（针对四层）
# class_means = calculate_class_mean_for_all_layers(activation_map_list_all_layers, label_list, num_classes)
#
# # 计算每个样本与其类别四层均值的相似度
# similarity_scores = [calculate_similarity_with_class_mean_for_all_layers(am_list, class_means[label])
#                      for am_list, label in zip(activation_map_list_all_layers, label_list)]
#
# # 结合 CDA 分数和同化程度，识别中毒样本
# combined_scores = identify_poisoned_samples_with_all_layers(sample_cda_scores, similarity_scores)
#
# # 打印结合后的最可疑样本
# for i in range(10):
#     data_index, label, predicted_class, combined_score = combined_scores[i]
#     print(f"Sample {i}: Index={data_index}, Label={label}, Predicted={predicted_class}, Combined Score={combined_score}")

