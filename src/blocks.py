import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

class CNNFE(nn.Module):
    def __init__(self):
        super().__init__()
    def get_backbone(self):
        raise NotImplementedError()
    def get_n_features(self):
        raise NotImplementedError()
    def forward(self, images):
        """
        Args:
            images: a tensor of dimension [batch, 3, height, width]
        Return:
            outputs: a tensor of dimension [batch, num_classes]
        """
        x = self.get_backbone()(images) # [batch, num_features, H', W'] = [batch, 2048, 1, 1]
        # x = torch.flatten(x, start_dim=1) # [batch, num_features]
        return x

class ResnetFE(CNNFE):
    version = {
        'resnet18':torchvision.models.resnet18,
        'resnet34':torchvision.models.resnet34,
        'resnet50':torchvision.models.resnet50,
        'resnet101':torchvision.models.resnet101,
        'resnet152':torchvision.models.resnet152
    }
    def __init__(self, version, feature_extract=True, pretrained=True):
        super(ResnetFE, self).__init__()
        resnet = ResnetFE.version[version](pretrained = pretrained)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        if feature_extract:
            for param in self.backbone.parameters():
                param.requires_grad = False
        self.n_features = resnet.fc.in_features
    def get_backbone(self):
        return self.backbone
    def get_n_features(self):
        return self.n_features
    def forward(self, images):
        """
        Args:
            images: a tensor of dimension [batch, 3, height, width]
        Return:
            outputs: a tensor of dimension [batch, num_classes]
        """
        x = self.backbone(images) # [batch, num_features, H', W'] = [batch, 2048, 1, 1]
        # x = torch.flatten(x, start_dim=1) # [batch, num_features]
        return x

class Inception(CNNFE):
    version = {
        'inception_v3':torchvision.models.inception_v3,
    }
    def __init__(self, version, feature_extract=True, pretrained=True):
        super(Inception, self).__init__()
        net = Inception.version[version](pretrained = pretrained)
        self.backbone = nn.Sequential(*list(net.children())[:-1])
        if feature_extract:
            for param in self.backbone.parameters():
                param.requires_grad = False
        self.n_features = net.fc.in_features
    def get_backbone(self):
        return self.backbone
    def get_n_features(self):
        return self.n_features
    def forward(self, images):
        """
        Args:
            images: a tensor of dimension [batch, 3, height, width]
        Return:
            outputs: a tensor of dimension [batch, num_classes]
        """
        x = self.backbone(images) # [batch, num_features, H', W'] = [batch, 2048, 1, 1]
        # x = torch.flatten(x, start_dim=1) # [batch, num_features]
        return x

class Mobilenet(CNNFE):
    version = {
        'mobilenet_v2':torchvision.models.mobilenet_v2,
        # 'mobilenet_v3_large': torchvision.models.mobilenet_v3_large,
        # 'mobilenet_v3_small': torchvision.models.mobilenet_v3_small,
    }
    def __init__(self, version, feature_extract=True, pretrained=True):
        super(Mobilenet, self).__init__()
        net = Mobilenet.version[version](pretrained = pretrained)
        self.backbone = nn.Sequential(*list(net.children())[:-1])
        if feature_extract:
            for param in self.backbone.parameters():
                param.requires_grad = False
        self.n_features = net.last_channel
    def get_backbone(self):
        return self.backbone
    def get_n_features(self):
        return self.n_features
    def forward(self, images):
        """
        Args:
            images: a tensor of dimension [batch, 3, height, width]
        Return:
            outputs: a tensor of dimension [batch, num_classes]
        """
        x = self.backbone(images) # [batch, num_features, H', W'] = [batch, 2048, 1, 1]
        # x = torch.flatten(x, start_dim=1) # [batch, num_features]
        return x

class EfficientNet(CNNFE):
    version = {
        'efficientnet_b0':torchvision.models.efficientnet_b0,
        'efficientnet_b1': torchvision.models.efficientnet_b1,
        'efficientnet_b2': torchvision.models.efficientnet_b2,
        'efficientnet_b3': torchvision.models.efficientnet_b3,
        'efficientnet_b4': torchvision.models.efficientnet_b4,
        'efficientnet_b5': torchvision.models.efficientnet_b5,
        'efficientnet_b6': torchvision.models.efficientnet_b6,
        'efficientnet_b7': torchvision.models.efficientnet_b7,

        # 'efficientnet_v2_s': torchvision.models.efficientnet_v2_s,
        # 'efficientnet_v2_m': torchvision.models.efficientnet_v2_m,
        # 'efficientnet_v2_l': torchvision.models.efficientnet_v2_l,
    }
    def __init__(self, version, feature_extract=True, pretrained=True):
        super(EfficientNet, self).__init__()
        net = EfficientNet.version[version](pretrained = pretrained)
        self.backbone = nn.Sequential(*list(net.children())[:-1])
        if feature_extract:
            for param in self.backbone.parameters():
                param.requires_grad = False
        self.n_features = net.classifier[1].in_features
    def get_backbone(self):
        return self.backbone
    def get_n_features(self):
        return self.n_features
    def forward(self, images):
        """
        Args:
            images: a tensor of dimension [batch, 3, height, width]
        Return:
            outputs: a tensor of dimension [batch, num_classes]
        """
        x = self.backbone(images) # [batch, num_features, H', W'] = [batch, 2048, 1, 1]
        # x = torch.flatten(x, start_dim=1) # [batch, num_features]
        return x

class SwinTransformer(CNNFE):
    version = {
        'swin_t':torchvision.models.swin_t,
        'swin_s': torchvision.models.swin_s,
        'swin_b': torchvision.models.swin_b,
        # 'swin_v2_t': torchvision.models.swin_v2_t,
        # 'swin_v2_s': torchvision.models.swin_v2_s,
        # 'swin_v2_b': torchvision.models.swin_v2_b,

    }
    def __init__(self, version, feature_extract=True, pretrained=True):
        super(SwinTransformer, self).__init__()
        net = SwinTransformer.version[version](weights = torchvision.models.Swin_B_Weights.IMAGENET1K_V1)
        print(net)
        self.backbone = nn.Sequential(*list(net.children())[:-1])
        for i, child in enumerate(list(net.children())):
            print('child ', i, child)
        print(net.head.in_features)
        # exit()
        if feature_extract:
            for param in self.backbone.parameters():
                param.requires_grad = False
        self.n_features = net.head.in_features
    def get_backbone(self):
        return self.backbone
    def get_n_features(self):
        return self.n_features
    def forward(self, images):
        """
        Args:
            images: a tensor of dimension [batch, 3, height, width]
        Return:
            outputs: a tensor of dimension [batch, num_classes]
        """
        x = self.backbone(images) # [batch, num_features, H', W'] = [batch, 2048, 1, 1]
        # x = torch.flatten(x, start_dim=1) # [batch, num_features]
        return 