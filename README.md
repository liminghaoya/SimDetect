## SimDetect: Detecting Backdoor Models via Assimilation Anomalies Analysis

![image-20240920112602314](README.assets/./image/1726803401941.jpg)

Pipeline for SimDetect. (1) Feature attention maps from benign, poisoned, and unknown models are analyzed using class assimilation rates. Frobenius norm calculations are employed to determine whether the unknown model is poisoned. (2) Covariance matrices and Covariance Discriminant Analysis (CDA) scores identify the top-10 suspect samples, revealing the poisoned category.



We used the methods integrated in [backdoor-toolbox](https://github.com/vtu81/backdoor-toolbox) for poisoning model training and metrics evaluation.

The specific training and evaluation methods are described in the links above.

## Dependency

This code was developed using torch==1.13.1+cu116. To set up the required environment, install PyTorch with CUDA manually, then by installing the other packages `pip install -r requirement.txt`.

### Preliminary

Datasets:

The original CIFAR10 and GTSRB datasets will be downloaded automatically.

Before any experiments, first initialise the clean retention and validation data using the command `python create_clean_set.py -dataset=$DATASET -clean_budget $N`, where `$DATASET = cifar10, gtsrb`

### Quick start

For example training a backdoor model that is attacked by BadNets

```
# Create a poisoned training set
python create_poisoned_set.py -dataset=cifar10 -poison_type=badnet -poison_rate=0.1
# Train on the poisoned training set
python train_on_poisoned_set.py -dataset=cifar10 -poison_type=badnet -poison_rate=0.1
# Test the backdoor model
python test_model.py -dataset=cifar10 -poison_type=badnet -poison_rate=0.1
```



SimDetect：

我们在./utils/vgg.py中展示了提取四层网络特征的示例

检测后门模型（同化程度高的模型往往更有可能为后门模型）

```
python cross_final.py
```

检测可疑类别（我们采用统计学的方法，输出的CDA值评分前十名中的可疑类别输出频次最高的定义为后门类别）

```
python defense_SimDetect.py
```

