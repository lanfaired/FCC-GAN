import torch
from torch import nn

#通道注意力
class channel_attention(nn.Module):
    def __init__(self, channel, ration=4):
        super(channel_attention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel//ration, bias=False),
            nn.ReLU(),
            nn.Linear(channel//ration, channel, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        avg_pool = self.avg_pool(x).view([b, c])
        max_pool = self.max_pool(x).view([b, c])

        avg_fc = self.fc(avg_pool)
        max_fc = self.fc(max_pool)

        out = self.sigmoid(max_fc+avg_fc).view([b, c, 1, 1])
        return x * out

#空间注意力
class spatial_attention(nn.Module):
    def __init__(self, kernel_size=3):
        super(spatial_attention, self).__init__()

        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size, stride=1,
                              padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_pool = torch.max(x, dim=1, keepdim=True).values
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        pool_out = torch.cat([max_pool, avg_pool], dim=1)
        conv = self.conv(pool_out)
        out = self.sigmoid(conv)

        return out * x

#将通道注意力和空间注意力进行融合
class CBAM(nn.Module):
    def __init__(self, channel, ration=4, kernel_size=3):
        super(CBAM, self).__init__()
        self.channel_attention = channel_attention(channel, ration)
        self.spatial_attention = spatial_attention(kernel_size)

    def forward(self, x):
        out = self.channel_attention(x)
        out = self.spatial_attention(out)
        return out



