import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
    

class SE_Block(nn.Module):
    def __init__(self, inchannel, ratio=4):
        super(SE_Block, self).__init__()
        # 全局平均池化(Fsq操作)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        # 两个全连接层(Fex操作)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels=inchannel, out_channels=inchannel // ratio, kernel_size=1, stride=1, bias=False),  # 从 c -> c/r
            nn.ReLU(),
            nn.Conv2d(in_channels=inchannel // ratio, out_channels=inchannel , kernel_size=1, stride=1, bias=False),  # 从 c/r -> c
            nn.Sigmoid()
        )
 
    def forward(self, x):
            y = self.gap(x)
            y = self.fc(y)
            return x * y.expand_as(x)


class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
 
        hidden_channels = oup // ratio
        new_channels = hidden_channels*(ratio-1)
 
        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, hidden_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )
 
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(hidden_channels, new_channels, dw_size, 1, dw_size//2, groups=hidden_channels, bias=False),
            nn.BatchNorm2d(new_channels)
        )
 
    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1,x2], dim=1)
        return out

    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, kernel_size=1, ratio=2):
        super(ResidualBlock, self).__init__()
        self.conv = GhostModule(inp=in_channels, oup=out_channels, kernel_size=kernel_size, ratio=ratio, dw_size=3, stride=stride, relu=True)
        self.se = SE_Block(inchannel=out_channels, ratio=4)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        self.relu = nn.Mish()
    
    def forward(self, x):
        identity = x
        x = self.conv(x)
        out = self.se(x)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = self.relu(out + identity)
        return out
 

class LWGN(nn.Module):
    def __init__(self, in_channels):
        super(LWGN, self).__init__()
        self.conv_init = nn.Conv2d(in_channels, 32, kernel_size=1, stride=1) # 224 -> 224
        # 编码器
        self.conv1 = nn.Sequential(ResidualBlock(32, 64, stride=2, ratio=2), ResidualBlock(64, 64, stride=1, kernel_size=3)) # 224 -> 112
        self.conv2 = nn.Sequential(ResidualBlock(64, 128, stride=2, ratio=2), ResidualBlock(128, 128, stride=1, kernel_size=3)) # 112 -> 56
        self.conv3 = nn.Sequential(ResidualBlock(128, 256, stride=2, ratio=2), ResidualBlock(256, 256, stride=1, kernel_size=3)) # 56 -> 28
        # 解码器
        self.pixel_shuffle = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1, groups=128),
                                             SE_Block(inchannel=128, ratio=4)) # 28 -> 56
        
        self.convT1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, groups=64),
                                            SE_Block(inchannel=64, ratio=4)) # 56 -> 112
        
        self.upconvT2 = nn.Sequential(nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1, groups=32), 
                                       SE_Block(inchannel=32, ratio=4)) # 112
        self.sa = SpatialAttention()
        # 输出层
        self.pos = nn.Sequential( nn.Conv2d(32, 1, 1), nn.Sigmoid())
        self.cos = nn.Sequential( nn.Conv2d(32, 1, 1))
        self.sin = nn.Sequential( nn.Conv2d(32, 1, 1))
        self.width = nn.Sequential( nn.Conv2d(32, 1, 1), nn.ReLU()) 

    def forward(self, x):
        x = self.conv_init(x)
        sa = self.sa(x)
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = self.pixel_shuffle(x)
        x = self.convT1(x)
        x = self.upconvT2(x)
        x = x * sa

        pos = self.pos(x)
        cos = self.cos(x)
        sin = self.sin(x)
        width = self.width(x)
        return pos, cos, sin, width
    
    def compute_loss(self, xc, yc):
        y_pos, y_cos, y_sin, y_width = yc
        pos_pred, cos_pred, sin_pred, width_pred = self(xc)

        p_loss = F.smooth_l1_loss(pos_pred, y_pos)
        cos_loss = F.smooth_l1_loss(cos_pred, y_cos)
        sin_loss = F.smooth_l1_loss(sin_pred, y_sin)
        width_loss = F.smooth_l1_loss(width_pred, y_width)

        return {
            'loss': p_loss + cos_loss+sin_loss + width_loss,
            'losses': {
                'p_loss': p_loss,
                'cos_loss': cos_loss,
                'sin_loss': sin_loss,
                'width_loss': width_loss
            },
            'pred': {
                'pos': pos_pred,
                'cos': cos_pred,
                'sin': sin_pred,
                'width': width_pred
            }
        }


if __name__ == '__main__':
    model = LWGN(3)
    model.to('cuda')
    x = torch.randn(1, 3, 224, 224).to('cuda')
    y, _, _, _ = model(x)
    summary(model, input_size=(3, 224, 224))