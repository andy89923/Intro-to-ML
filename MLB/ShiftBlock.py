import torch
from mmcv.cnn import constant_init, kaiming_init
from torch import nn

class ShiftBlock(nn.Module):
    def __init__(self, dim, n_frames=8, _bn_mom=1e-2, _bn_eps=1e-3, TEA=True, TEA_ratio=4):
        super(ShiftBlock, self).__init__()
        self.shift_dim = dim
        self.n_frames = n_frames
        self._bn_mom = _bn_mom
        self._bn_eps = _bn_eps

        self.TEA = TEA
        self.TEA_ratio = TEA_ratio

        # Reference:
        # Efficient Temporal-Spatial Feature Grouping For Video Action Recognition  
        # (https://ieeexplore.ieee.org/abstract/document/9190997)
        # Grouped Spatial-Temporal Aggregation for Efficient Action Recognition 
        # (https://arxiv.org/abs/1909.13130)
        self.shift_groups = dim
        
        # Learnable Gated Temporal Shift Module
        # Reference:
        # Learnable Gated Temporal Shift Module for Deep Video Inpainting 
        # (https://arxiv.org/abs/1907.01131)
        self._shiftConv_1 = nn.Conv1d(self.shift_dim, self.shift_dim, 3, groups = self.shift_groups, stride = 1, padding = 1, bias=True)

        # Gated Convolution
        # Reference:
        # Free-Form Image Inpainting with Gated Convolution 
        # (https://arxiv.org/abs/1806.03589)
        self._gateConv_1 = nn.Conv1d(self.shift_dim, self.shift_dim, 3, groups = self.shift_groups, stride=1, padding=1, bias=True)
        self._gateConv_bn = nn.BatchNorm1d(num_features=self.shift_dim, momentum=self._bn_mom, eps=self._bn_eps)
        self._gateActivation = nn.Sigmoid()
#         nn.init.kaiming_normal_(self._shiftConv_1.weight, a=0, mode='fan_in')
#         nn.init.kaiming_normal_(self._gateConv_1.weight, a=0, mode='fan_in')

        # TEA
        # Reference:
        # TEA: Temporal Excitation and Aggregation for Action Recognition 
        # (https://arxiv.org/abs/2004.01398)
        if TEA:
            self.TEA_GAP = nn.AdaptiveAvgPool2d(1)
            self.TEA_squeeze_conv = nn.Conv2d(self.shift_dim, self.shift_dim//self.TEA_ratio, 1, stride=1, padding=0, bias=True)
            self.TEA_expand_conv = nn.Conv2d(self.shift_dim//self.TEA_ratio, self.shift_dim, 1, stride=1, padding=0, bias=True)
            self.TEA_activation = nn.Sigmoid()
        
    def MotionDiffConv(self, x):
        NT, C, H, W = x.shape
        T = self.n_frames
        N = NT//T
        
        x = x.reshape(N, T, -1, H, W)
        
        xt = x[:, :-1]
        xt_1 = x[:, :1]
        
        diff = xt_1-xt
        zero_pad = torch.zeros((N, 1, C, H, W)).cuda()
        diff = torch.cat((diff, zero_pad), 1)
        
        diff = diff.reshape(NT, -1, H, W)
        x = x.reshape(NT, -1, H, W)
        
        out = self.TEA_GAP(diff)
        out = self.TEA_expand_conv(out)
        out = self.TEA_activation(out)
        
        return out
    
    def MotionExcitation(self, x):
        NT, C, H, W = x.shape
        residual = x
        x = self.TEA_squeeze_conv(x)
        x = self.MotionDiffConv(x)
        out = residual*x+residual

        return out
        
    
    def LearnableTemporalShiftModule(self, x):
        NT, C, H, W = x.shape
        N = NT//self.n_frames
        T = self.n_frames
        
        S = x
        
        if self.TEA:
            S = self.MotionExcitation(S)
        
        x = x.reshape(N, T, -1, H, W)
        S = S.reshape(N, T, -1, H, W)
        
        # N, T, C, H, W
        x = x.permute(0, 3, 4, 2, 1)
        S = S.permute(0, 3, 4, 2, 1)
        # N, H, W, C, T
        x = x.reshape(-1, C, T)
        S = S.reshape(-1, C, T)
        
        G = x
        
        # NHW, C, T
        S = self._shiftConv_1(S)
    
        # NHW, C, T
        G = self._gateConv_1(G)
        G = self._gateConv_bn(G)
        G = self._gateActivation(G)
    
        out = S*G
        out = out.reshape(N, H, W, -1, T)
        # N, H, W, C, T
        out = out.permute(0, 4, 3, 1, 2)
        # N, T, C, H, W
        out = out.reshape(N*T, -1, H, W)
        
        return out

    def forward(self, x):
        out = self.LearnableTemporalShiftModule(x)
        return out
