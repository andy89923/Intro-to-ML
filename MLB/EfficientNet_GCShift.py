import torch
import torch.nn as nn
# from efficientnet_pytorch import EfficientNet
from EfficientNetModel import EfficientNet
import numpy as np

class EfficientNet_GCShift(nn.Module):
    def __init__(self, device, num_segments, batch_size):
        super(EfficientNet_GCShift, self).__init__()
        self.device = device
        self.num_segments = num_segments
        self.batch_size = batch_size

        self.model = EfficientNet.from_pretrained('efficientnet-b0', advprop=True, include_top=False).cuda()
        print(self.model)
                        
        self.ImageGlobalAvgPool = nn.AdaptiveAvgPool2d(1)

        self.fc_layers = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1536, 101),
        )
        
    def forward(self, x):
        N, T, C, H, W = x.size()
        x = x.reshape(N*T, C, H, W)
        x = self.model(x)

        _, _, H, W = x.size()
        x = x.reshape(N, T, -1, H, W)
        
        # N, T, C, H, W
        N, T, C, H, W = x.shape
        x = self.ImageGlobalAvgPool(x)
        return self.fc_layers(x.flatten(1))