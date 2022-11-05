import timm
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
    
class EfficientNetB3DSPlus(nn.Module):
    def __init__(self, model_name, n_class=2, pretrained=True):
        super().__init__()
        backbone = timm.create_model(model_name, pretrained=pretrained)
        try:
            n_features = backbone.classifier.in_features
        except:
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