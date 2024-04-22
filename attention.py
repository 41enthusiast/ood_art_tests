import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Attention blocks
Reference: Learn To Pay Attention
"""
class ProjectorBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ProjectorBlock, self).__init__()
        self.op = nn.Conv2d(in_channels=in_features, out_channels=out_features,
            kernel_size=1, padding=0, bias=False)#1x1 conv, op: bsz,out_ch,h,w

    def forward(self, x):
        return self.op(x)


class SpatialAttn(nn.Module):
    def __init__(self, in_features, normalize_attn=True):
        super(SpatialAttn, self).__init__()
        self.normalize_attn = normalize_attn
        self.op = nn.Conv2d(in_channels=in_features, out_channels=1,
            kernel_size=1, padding=0, bias=False)

    def forward(self, l, g):
        N, C, H, W = l.size()
        c = self.op(l+g) # (batch_size,1,H,W)
        # compute the attention map
        if self.normalize_attn:
            a = F.softmax(c.view(N,1,-1), dim=2).view(N,1,H,W)#softmax sum to 1 in the feature dim(hxw)
        else:
            a = torch.sigmoid(c)
        g = torch.mul(a.expand_as(l), l)#reweight the local features
        if self.normalize_attn:
            g = g.view(N,C,-1).sum(dim=2) # (batch_size,C)
        else:
            g = F.adaptive_avg_pool2d(g, (1,1)).view(N,C)
        return c.view(N,1,H,W), g


# Test
if __name__ == '__main__':
    # 2d block
    spatial_block = SpatialAttn(in_features=3)
    l = torch.randn(16, 3, 128, 128)
    g = torch.randn(16, 3, 128, 128)
    print(spatial_block(l, g))