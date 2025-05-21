import torch
import torch.nn as nn
import torchvision.models as models

class DualResNetWithMetadata(nn.Module):
    def __init__(self, tabular_dim, num_classes):
        super().__init__()
        self.resnet_sentinel = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.resnet_osm = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.resnet_sentinel.fc = nn.Identity()
        self.resnet_osm.fc = nn.Identity()

        self.mlp = nn.Sequential(
            nn.Linear(tabular_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(2048 * 2 + 32, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, img_sentinel, img_osm, tabular):
        feat_sen = self.resnet_sentinel(img_sentinel)
        feat_osm = self.resnet_osm(img_osm)
        feat_tab = self.mlp(tabular)
        combined = torch.cat([feat_sen, feat_osm, feat_tab], dim=1)
        return self.classifier(combined)
