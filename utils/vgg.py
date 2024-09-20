'''
Modified from https://github.com/pytorch/vision.git
'''
import math

import torch.nn as nn
import torch.nn.init as init

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


class VGG(nn.Module):
    '''
    VGG model
    '''

    #ori
    # def __init__(self, features, num_classes=10):
    #     super(VGG, self).__init__()
    #     self.features = features
    #     self.classifier = nn.Sequential(
    #         nn.Dropout(),
    #         nn.Linear(512, 512),
    #         nn.ReLU(True),
    #         nn.Dropout(),
    #         nn.Linear(512, 512),
    #         nn.ReLU(True),
    #         nn.Linear(512, num_classes),
    #     )
    #     # Initialize weights
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #             m.weight.data.normal_(0, math.sqrt(2. / n))
    #             m.bias.data.zero_()
    #
    #
    # def forward(self, x, return_hidden=False):
    #
    #     x = self.features(x)
    #     x = x.view(x.size(0), -1)
    #     if return_hidden:
    #         hidden = x
    #     x = self.classifier(x)
    #     if return_hidden:
    #         return x, hidden
    #     else:
    #         return x

    def __init__(self, features, num_classes=10):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, num_classes),
        )
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def from_input_to_features(self, x):
        # Pass the input through the feature extractor part of the model
        x = self.features(x)
        # Flatten the output from the feature extractor (since VGG usually flattens before feeding into the classifier)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        return x  # Return the features

    def from_features_to_output(self, features):
        # 传递特征给分类器部分并返回输出
        return self.classifier(features)

    def forward(self, x, return_hidden=False, return_activation=False):

        activation1 = None
        activation2 = None
        activation3 = None
        idx_num=0
        layer_num=0

        # Assuming features are defined as a Sequential model, we can track activations
        for idx, layer in enumerate(self.features):
            idx_num+=1
            layer_num+=1
            x = layer(x)
            if idx == 5:  # Example: After 10th layer
                activation1 = x
            elif idx == 15:  # Example: After 25th layer
                activation2 = x
            elif idx == 30:  # Example: After 40th layer
                activation3 = x
            elif idx == 40:  # Example: After 40th layer
                activation4 = x
        # print('idx_num',idx_num)


        x = x.view(x.size(0), -1)
        hidden = x
        x = self.classifier(x)
        out = x

        if return_hidden:
            return out, hidden
        elif return_activation:
            return out, activation1, activation2, activation3,activation4
        else:
            return out

    def make_layers(cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def freeze_feature(self):
        for name, para in self.named_parameters():
            if name.count('classifier') == 0: # non-linear layer
                para.requires_grad = False
    
    def unfreeze_feature(self):
        for name, para in self.named_parameters():
            if name.count('classifier') == 0: # non-linear layer
                para.requires_grad = True

    def freeze_fc(self):
        for name, para in self.named_parameters():
            if name.count('classifier') > 0: # non-linear layer
                para.requires_grad = False
                
    def freeze_fc(self):
        for name, para in self.named_parameters():
            if name.count('classifier') > 0: # non-linear layer
                para.requires_grad = True

    def partial_forward(self,x):
        x = self.features(x)
        x = x.view(x.size(0), -1)

        partial_classifier = self.classifier[:-1]
        x = partial_classifier(x)
        return x


class VGG_low_dim(nn.Module):
    '''
    VGG model
    '''
    def __init__(self, features, num_classes=10):
        super(VGG_low_dim, self).__init__()

        self.features = features

        self.reducer = nn.Linear(512, 8)

        self.low_dim_classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(8, 8),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(8, 8),
            nn.ReLU(True),
            nn.Linear(8, num_classes)
        )
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()


    def forward(self, x, return_hidden=False):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.reducer(x)

        if return_hidden:
            hidden = x
        x = self.low_dim_classifier(x)
        if return_hidden:
            return x, hidden
        else:
            return x

    def partial_forward(self,x):
        x = self.features(x)
        x = x.view(x.size(0), -1)

        partial_classifier = self.classifier[:-1]
        x = partial_classifier(x)
        return x




def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 
          512, 512, 512, 512, 'M'],
}


def vgg11():
    """VGG 11-layer model (configuration "A")"""
    return VGG(make_layers(cfg['A']))


def vgg11_bn():
    """VGG 11-layer model (configuration "A") with batch normalization"""
    return VGG(make_layers(cfg['A'], batch_norm=True))


def vgg13():
    """VGG 13-layer model (configuration "B")"""
    return VGG(make_layers(cfg['B']))


def vgg13_bn():
    """VGG 13-layer model (configuration "B") with batch normalization"""
    return VGG(make_layers(cfg['B'], batch_norm=True))


def vgg16():
    """VGG 16-layer model (configuration "D")"""
    return VGG(make_layers(cfg['D']))


def vgg16_bn(num_classes=10):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    return VGG(make_layers(cfg['D'], batch_norm=True), num_classes=num_classes)

def vgg16_low_dim_bn(num_classes=10):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    return VGG_low_dim(make_layers(cfg['D'], batch_norm=True), num_classes=num_classes)


def vgg19():
    """VGG 19-layer model (configuration "E")"""
    return VGG(make_layers(cfg['E']))


def vgg19_bn():
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    return VGG(make_layers(cfg['E'], batch_norm=True))
