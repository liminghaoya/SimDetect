from utils import resnet, vgg, mobilenetv2, ember_nn, gtsrb_cnn, wresnet    # backdoor-tool-box中的库，用于构建模型
from utils import supervisor, tools                                         # backdoor-tool-box中的库，用于各种指标计算
import config                                                               # backdoor-tool-box中的库，加载模型和攻击的各种配置参数
import os, sys
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import nn
from torchvision import datasets, transforms
import torch
from PIL import Image


## 计算 attention map
#原始p设置为2
def attention_map(fm, p=2.5, eps=1e-6):
    am = torch.pow(torch.abs(fm), p)
    am = torch.sum(am, dim=1, keepdim=True)
    norm = torch.norm(am, dim=(2, 3), keepdim=True)
    am = torch.div(am, norm + eps)
    am = F.interpolate(am, size=(32, 32), mode='bilinear', align_corners=False)
    return am

class VGGWithAttention(nn.Module):
    def __init__(self, original_vgg):
        super(VGGWithAttention, self).__init__()
        self.features = nn.Sequential(
            *list(original_vgg.features.children())[:16],  # 插入注意力模块之前的层
            AttentionModule(256),  # 插入注意力模块
            *list(original_vgg.features.children())[16:],  # 插入注意力模块之后的层
        )
        self.classifier = original_vgg.classifier

    def forward(self, x):
        x, attention = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x, attention  # 返回注意力图


## 数据集加载以及预处理。为使每次绘制结果一致，关闭翻转和裁切操作(此处为backdoor-tool-box的示例，使用其他代码仓库时需要进行替换)
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
kwargs = {'pin_memory': True}

data_transform_aug = transforms.Compose([
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomCrop(32, 4),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
])

data = torch.load("/root/root/lmh/backdoor-toolbox_ori/poisoned_train_set/cifar10/badnet_0.100_poison_seed=0/imgs")
label = torch.load("/root/root/lmh/backdoor-toolbox_ori/poisoned_train_set/cifar10/badnet_0.100_poison_seed=0/labels")

poisoned_data_path = "/root/root/lmh/backdoor-toolbox_ori/poisoned_train_set/cifar10/badnet_0.100_poison_seed=0/imgs"
poisoned_label_path = "/root/root/lmh/backdoor-toolbox_ori/poisoned_train_set/cifar10/badnet_0.100_poison_seed=0/labels"

poisoned_set = tools.IMG_Dataset(data_dir=poisoned_data_path, label_path=poisoned_label_path,
                                 transforms=data_transform_aug)
poisoned_set_loader = torch.utils.data.DataLoader(poisoned_set, batch_size=256, shuffle=False,
                                                  worker_init_fn=tools.worker_init, **kwargs)

#47,61,66,98,117,156,221  gtsrb中毒index
#6,58,80,124,182，213   cifar10中毒index
## 获取要绘制注意力的图片（此处可以选一张图片直接进行预处理，不一定需要dataloader）
for data, target in poisoned_set_loader:
    img = data[58].unsqueeze(0).cuda()
    print("imageshape", img.size())
    #保存这张ori图像为png
    img_to_save = data[58]
    break

# 将张量格式的图像转换为PIL格式
img_to_save = img_to_save.cpu().squeeze(0)
img_to_save = img_to_save.permute(1, 2, 0)  # 将CHW转换为HWC格式
img_to_save = img_to_save * torch.tensor([0.247, 0.243, 0.261]) + torch.tensor([0.4914, 0.4822, 0.4465])  # 反归一化
img_to_save = (img_to_save * 255).byte()  # 转换为8位图像
img_to_save = Image.fromarray(img_to_save.numpy())

# 保存图像为PNG格式
img_to_save.save("attention_map/ori_img.png")

## 模型加载（此处为backdoor-tool-box的示例，使用其他代码仓库时需要进行替换）
# module = resnet.ResNet50()
module = vgg.vgg16_bn()
module.load_state_dict(torch.load("/root/root/lmh/backdoor-toolbox_ori/poisoned_train_set/cifar10/badnet_0.100_poison_seed=0/full_base_aug_seed=2333.pt"))
module = nn.DataParallel(module)

# 初始化模型
module = VGGWithAttention(module)

module = module.cuda()
module.eval()


## 画am
output, activation1, activation2, activation3,activation4 = module.forward(img, return_activation=True)    #需要修改模型的forward计算
am1 = attention_map(activation1)
am2 = attention_map(activation2)
am3 = attention_map(activation3)
am4 = attention_map(activation4)

print("output:", output)
print("output after softmax:" ,F.softmax(output))
# plt.imshow(am4[0].squeeze().cpu().detach().numpy(), cmap=plt.cm.hot, vmin=0, vmax=1)

ourdefense=True
if ourdefense:
    plt.imshow(am1.squeeze().cpu().detach().numpy(), cmap='jet')
    # plt.savefig('attention_map/test.png')
    plt.savefig('attention_map/benignmodel_test1.png')

    plt.imshow(am2.squeeze().cpu().detach().numpy(), cmap='jet')
    # plt.savefig('attention_map/test.png')
    plt.savefig('attention_map/benignmodel_test2.png')

    plt.imshow(am3.squeeze().cpu().detach().numpy(), cmap='jet')
    # plt.savefig('attention_map/test.png')
    plt.savefig('attention_map/benignmodel_test3.png')

    plt.imshow(am4.squeeze().cpu().detach().numpy(), cmap='jet')
    # plt.savefig('attention_map/test.png')
    plt.savefig('attention_map/benignmodel_test4.png')
else:
    plt.imshow(am1.squeeze().cpu().detach().numpy(), cmap='jet')
    # plt.savefig('attention_map/test.png')
    plt.savefig('attention_map/BadExpert_test1.png')

    plt.imshow(am2.squeeze().cpu().detach().numpy(), cmap='jet')
    # plt.savefig('attention_map/test.png')
    plt.savefig('attention_map/BadExpert_test2.png')

    plt.imshow(am3.squeeze().cpu().detach().numpy(), cmap='jet')
    # plt.savefig('attention_map/test.png')
    plt.savefig('attention_map/BadExpert_test3.png')

    plt.imshow(am4.squeeze().cpu().detach().numpy(), cmap='jet')
    # plt.savefig('attention_map/test.png')
    plt.savefig('attention_map/BadExpert_test4.png')