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
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(n_features, n_class)

    def forward_features(self, x):
        x = self.backbone(x)
        return x

    # def forward(self, x):
    #     x = self.forward_features(x)
    #     # x = self.pool(x).view(x.size(0), -1)
    #     x = self.dropout(x)
    #     x = x.mean(dim=1)
    #     x = self.classifier(x)
    #     return x83794
    
    def forward(self, x):
        feats = self.forward_features(x)
        x = self.pool(feats).view(x.size(0), -1)
        x = self.dropout(x)
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

        if self.multi_drop:
            for i, dropout in enumerate(self.head_drops):
                if i == 0:
                    output = self.head(dropout(h))
                else:
                    output += self.head(dropout(h))
            output /= len(self.head_drops)
        else:
            output = self.head(h)
        return output

    def forward(self, x):
        x = self.model(x)
        return x

class SwinTransformerv2(nn.Module):
    def __init__(self, model_name, n_class=2, pretrained=True):
        super().__init__()
        backbone = timm.create_model(model_name, pretrained=pretrained)
        n_features = backbone.head.in_features
        self.backbone = backbone
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(n_features, n_class)

    def forward_features(self, x):
        x = self.backbone.forward_features(x)
        return x

    def forward(self, x):
        feats = self.forward_features(x)
        x = feats.mean(dim=1)
        x = self.classifier(x)
        return x

class SwinTransformer(nn.Module):
    def __init__(self, model_name, n_class=2, pretrained=True, drop_rate=0):
        super().__init__()
        backbone = timm.create_model(model_name, pretrained=pretrained, drop_rate=drop_rate)
        n_features = backbone.head.in_features
        self.backbone = backbone
        # print(self.backbone)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(n_features, n_class)


    def forward_features(self, x):
        x = self.backbone.forward_features(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.dropout(x)
        x = x.mean(dim=1)
        x = self.classifier(x)
        return x
    
