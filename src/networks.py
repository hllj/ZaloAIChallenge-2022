import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from .blocks import fe
class CNNModel(nn.Module):
    def __init__(self, fe_name, version, feature_extract=True, pretrained=True, number_class=2, drop_p=0.3):
        super(CNNModel, self).__init__()
        fe_module = getattr(fe, fe_name)
        self.backbone = fe_module(version, feature_extract=True, pretrained=True)
        self.n_features = self.backbone.get_n_features()
        self.fc_cnn = nn.Sequential(
            nn.Linear(self.n_features, 512),
            nn.BatchNorm1d(512, momentum=0.01),
            nn.ReLU(),
            nn.Dropout(p=drop_p),
            nn.Linear(512, number_class)
        )

    def forward(self, images):
        """
        Args:
            images: a tensor of dimension [batch, 3, height, width]
        Return:
            outputs: a tensor of dimension [batch, num_classes]
        """
        self.backbone.eval()
        with torch.no_grad():
            print(images.shape)
            x = self.backbone(images) # [batch, num_features, H', W'] = [batch, 2048, 1, 1]
            x = torch.flatten(x, start_dim=1) # [batch, num_features]
        x = self.fc_cnn(x)
        return 