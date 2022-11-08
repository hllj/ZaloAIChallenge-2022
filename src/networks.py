import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class EfficientNetB3DSPlus(nn.Module):
    def __init__(self, model_name, n_class=2, pretrained=True):
        super().__init__()
        backbone = timm.create_model(model_name, pretrained=pretrained)
        try:
            n_features = backbone.classifier.in_features
        except Exception as ex:
            print("exception", ex)
            n_features = backbone.fc.in_features
        self.backbone = nn.Sequential(*backbone.children())[:-2]
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(n_features, n_class)

    def forward_features(self, x):
        x = self.backbone(x)
        return x

    def forward(self, x):
        feats = self.forward_features(x)
        x = self.pool(feats).view(x.size(0), -1)
        x = self.classifier(x)
        return x


class ResNext(nn.Module):
    def __init__(self, model_name, n_class=2, pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        n_features = self.model.fc.in_features
        self.model.fc = nn.Linear(n_features, n_class)

    def forward(self, x):
        x = self.model(x)
        return x


class VITBase(nn.Module):
    def __init__(self, model_name, n_class=2, pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        n_features = self.model.head.in_features
        self.model.head = nn.Linear(n_features, n_class)

    def forward(self, x):
        x = self.model(x)
        return x


class DeitBase(nn.Module):
    def __init__(self, model_name, n_class=2, pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        n_features = self.model.head.in_features
        self.model.head = nn.Linear(n_features, n_class)

    def forward(self, x):
        x = self.model(x)
        return x
