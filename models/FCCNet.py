import time

from ptflops import get_model_complexity_info
from torch import nn, Tensor
import torch
from torch.nn import init
from torchvision.models.mobilenetv2 import _make_divisible
import torch.nn.functional as F

class Trans_Up(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Trans_Up, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_planes, out_planes, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_planes),
            nn.GELU(),
        ]
        self.convTrans = nn.Sequential(*layers)

    def forward(self, x):
        out = self.convTrans(x)
        return out


class GC(nn.Module):
    def __init__(self, in_channel):
        super(GC, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel,in_channel,kernel_size=1,stride=1,padding=0),
            nn.GELU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=1, stride=1, padding=0),
            nn.GELU()
        )

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        return x_1 * x_2


class LCB(nn.Module):
    def __init__(self, in_channel):
        super(LCB, self).__init__()
        # 填充边缘 保证边缘细节处理
        self.conv_p = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=1, padding=1, stride=1, padding_mode='reflect'),
            nn.AvgPool2d(kernel_size=3,stride=1,padding=0),
            nn.Conv2d(in_channel, in_channel, 1, groups=in_channel, bias=False)
        )
        # 通道权重
        self.conv_c = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channel, in_channel, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        self.chw_w = nn.Sequential(
            nn.Conv2d(in_channel, 1, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_p = self.conv_p(x)
        xc_weight = self.conv_c(x)
        chw_weight = self.chw_w(xc_weight * x_p)
        return chw_weight * x


class Color_Comp(nn.Module):
    def __init__(self, in_channel):
        super(Color_Comp, self).__init__()

        self.color = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 3, padding=1, bias=False),
            GC(in_channel),
            LCB(in_channel)
        )

    def forward(self, x):
        c_x = self.color(x)
        return c_x + x

# Multi-scale Dense Block Feature
class MSDB(nn.Module):
    def __init__(self, block_num, inter_channel, channel):
        super(MSDB, self).__init__()
        concat_channels = channel + block_num * inter_channel
        channels_now = channel
        self.group_list = nn.ModuleList([])
        for i in range(block_num):
            group = nn.Sequential(
                nn.Conv2d(in_channels=channels_now, out_channels=inter_channel, kernel_size=3,
                          stride=1, padding=1),
                nn.GELU(),
            )
            self.add_module(name='group_%d' % i, module=group)
            self.group_list.append(group)
            channels_now += inter_channel

        assert channels_now == concat_channels
        self.fusion = nn.Sequential(
            nn.Conv2d(concat_channels, channel, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
        )

        self.conv_3_1 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=1,padding=1,bias=True)
        self.conv_5_1 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=5, stride=1,padding=2,bias=True)
        self.conv_f = nn.Conv2d(in_channels=channel * 2, out_channels=channel, kernel_size=3, stride=1, padding=1,bias=True)
        self.relu = nn.GELU()

    def forward(self, x):
        x_3_1 = self.relu(self.conv_3_1(x))
        x_5_1 = self.relu(self.conv_5_1(x))
        x_3 = x_3_1 + x
        x_5 = x_5_1 + x
        x_f = torch.cat([x_3, x_5], dim=1)
        x_f = self.relu(self.conv_f(x_f))

        feature_list = [x_f]
        for group in self.group_list:
            inputs = torch.cat(feature_list, dim=1)
            outputs = group(inputs)
            feature_list.append(outputs)
        inputs = torch.cat(feature_list, dim=1)
        fusion_outputs = self.fusion(inputs)
        block_outputs = fusion_outputs + x
        return block_outputs


class Trans_Down(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Trans_Down, self).__init__()
        self.conv0 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=2, padding=1)
        self.IN1 = nn.InstanceNorm2d(out_planes)
        self.relu = nn.GELU()

    def forward(self, x):
        out = self.relu(self.IN1(self.conv0(x)))
        return out


class ffconv(nn.Module):
    def __init__(self, inchannel, kernel_size=1):
        super(ffconv, self).__init__()
        channel_grow = inchannel // 2
        self.conv = nn.Sequential(
            nn.Conv2d(inchannel, channel_grow, kernel_size=kernel_size, padding=(kernel_size - 1) // 2),
            nn.InstanceNorm2d(channel_grow, affine=True),
            nn.GELU()
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(channel_grow + inchannel, inchannel, kernel_size=kernel_size, padding=(kernel_size - 1) // 2),
            nn.InstanceNorm2d(inchannel, affine=True),
            nn.GELU()
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(inchannel*2, inchannel, 1, 1, 0),
            nn.InstanceNorm2d(inchannel, affine=True),
            nn.GELU(),
        )

    def forward(self, x):
        out = self.conv(x)
        out = torch.cat((x, out), 1)
        out = self.conv1(out)
        out = torch.cat((x, out), 1)
        out = self.fusion(out)
        return out


class FLU(nn.Module):
    def __init__(self,in_channel):
        super(FLU,self).__init__()

        self.ffconv1 = ffconv(in_channel)
        self.ffconv2 = ffconv(in_channel)

    def forward(self,x):
        _, _, H, W = x.shape
        fft = torch.fft.rfft2(x, norm='backward')
        mag = torch.abs(fft)
        pha = torch.angle(fft)

        mag_f = self.ffconv1(mag)
        pha_f = self.ffconv2(pha)

        real = mag_f * torch.cos(pha_f)
        imag = mag_f * torch.sin(pha_f)
        x_out = torch.complex(real, imag)
        out = torch.fft.irfft2(x_out, s=(H, W), norm='backward')

        return out + x


class FAsyCA(nn.Module):
    def __init__(self, in_channel,out_channel):
        super(FAsyCA, self).__init__()
        self.down_1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1,stride=1),
            nn.InstanceNorm2d(out_channel),
            nn.GELU(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        self.flu_l = FLU(in_channel)
        self.flu_g = FLU(in_channel)

        self.conv1 = ffconv(in_channel)
        self.conv2 = ffconv(in_channel)

        self.conv_t = nn.Sequential(
            nn.Conv2d(out_channel,out_channel,kernel_size=1,padding=0,stride=1),
            nn.InstanceNorm2d(out_channel),
            nn.GELU(),
            nn.Conv2d(out_channel, out_channel, kernel_size=1, padding=0, stride=1),
            nn.InstanceNorm2d(out_channel),
            nn.GELU(),
        )

    def forward(self, x):
        x_d = self.down_1(x)
        xl, xg = torch.chunk(x_d,chunks=2,dim=1)
        xl_f = self.flu_l(xl)
        xg_f = self.flu_g(xg)

        xl_c = self.conv1(xl)
        xg_c = self.conv2(xg)

        xl_t = xl_f + xg_c
        xg_t = xl_c + xg_f

        x_t = torch.cat([xl_t, xg_t],dim=1)
        out = self.conv_t(x_t)
        return out + x_d


class Encoder_A(nn.Module):
    def __init__(self):
        super(Encoder_A, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.InstanceNorm2d(16, affine=True),
            nn.GELU(),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.InstanceNorm2d(16, affine=True),
            nn.GELU(),
        )
        self.Dense_Down_1 = MSDB(3, 32, 32)
        self.Dense_Down_2 = MSDB(3, 64, 64)
        self.Dense_Down_3 = MSDB(3, 128, 128)
        self.Dense_Down_4 = MSDB(3, 256, 256)
        # 下采样
        self.trans_down_1 = Trans_Down(16, 32)
        self.trans_down_2 = Trans_Down(32, 64)
        self.trans_down_3 = Trans_Down(64, 128)
        self.trans_down_4 = Trans_Down(128, 256)

    def forward(self, x):
        down_11 = self.conv1(x)              # 16 256 256

        down_1 = self.trans_down_1(down_11)  # 32 128 128
        down_21 = self.Dense_Down_1(down_1)  # 32 128 128

        down_2 = self.trans_down_2(down_21)  # 64 64 64
        down_31 = self.Dense_Down_2(down_2)  # 64 64 64

        down_3 = self.trans_down_3(down_31)  # 128 32 32
        down_41 = self.Dense_Down_3(down_3)  # 128 32 32

        down_4 = self.trans_down_4(down_41)  # 256 16 16
        down_51 = self.Dense_Down_4(down_4)  # 256 16 16

        return [down_51, down_41, down_31, down_21, down_11]

class Encoder_U(nn.Module):
    def __init__(self):
        super(Encoder_U, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.InstanceNorm2d(16, affine=True),
            nn.Hardswish(inplace=True),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.InstanceNorm2d(16, affine=True),
            nn.Hardswish(inplace=True),
        )
        self.Dense_Down_1 = MSDB(3, 32, 32)
        self.Dense_Down_2 = MSDB(3, 64, 64)
        self.Dense_Down_3 = MSDB(3, 128, 128)
        self.Dense_Down_4 = MSDB(3, 256, 256)
        # 下采样
        self.trans_down_1 = Trans_Down(16, 32)
        self.trans_down_2 = Trans_Down(32, 64)
        self.trans_down_3 = Trans_Down(64, 128)
        self.trans_down_4 = Trans_Down(128, 256)

    def forward(self, x):
        down_11 = self.conv1(x)              # 16 256 256

        down_1 = self.trans_down_1(down_11)  # 32 128 128
        down_21 = self.Dense_Down_1(down_1)  # 32 128 128

        down_2 = self.trans_down_2(down_21)  # 64 64 64
        down_31 = self.Dense_Down_2(down_2)  # 64 64 64

        down_3 = self.trans_down_3(down_31)  # 128 32 32
        down_41 = self.Dense_Down_3(down_3)  # 128 32 32

        down_4 = self.trans_down_4(down_41)  # 256 16 16
        down_51 = self.Dense_Down_4(down_4)  # 256 16 16

        return [down_51, down_41, down_31, down_21, down_11]


class SCAM(nn.Module):
    def __init__(self, input_c: int, squeeze_factor: int = 4):
        super(SCAM, self).__init__()
        squeeze_c = _make_divisible(input_c // squeeze_factor, 8)
        self.fc1 = nn.Conv2d(input_c, squeeze_c, 1)
        self.fc2 = nn.Conv2d(squeeze_c, input_c, 1)

    def forward(self, x: Tensor) -> Tensor:
        scale = F.adaptive_avg_pool2d(x, output_size=(1, 1))
        scale = self.fc1(scale)
        scale = F.relu(scale, inplace=True)
        scale = self.fc2(scale)
        scale = F.hardsigmoid(scale, inplace=True)
        return scale * x


class MSA(nn.Module):
    def __init__(self, in_channels, out_channels):

        super(MSA, self).__init__()

        self.conv_3_1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1,
                                  bias=True)
        self.conv_5_1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=5, stride=1, padding=2,
                                  bias=True)
        self.conv_7_1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=7, stride=1, padding=3,
                                  bias=True)
        self.conv_1_1 = nn.Conv2d(in_channels=in_channels * 2, out_channels=out_channels, kernel_size=1, stride=1, padding=0,
                                   bias=True)
        self.conv_1_2 = nn.Conv2d(in_channels=in_channels * 3, out_channels=out_channels, kernel_size=1, stride=1, padding=0,
                                  bias=True)
        self.cam = SCAM(out_channels)
        self.relu = nn.GELU()

    def forward(self, x):
        output_3_1 = self.relu(self.conv_3_1(x))
        output_5_1 = self.relu(self.conv_5_1(x))
        output_7_1 = self.relu(self.conv_7_1(x))
        output1 = torch.cat([output_3_1, output_5_1, output_7_1], 1)
        output6 = self.relu(self.conv_1_2(output1))
        output2 = self.cam(output6)
        output3 = torch.cat([output2, output6], 1)
        output4 = self.conv_1_1(output3)
        output = output4 + x
        return output


class Generator_A(nn.Module):
    def __init__(self):
        super(Generator_A, self).__init__()

        self.F_AsyCA_1 = FAsyCA(16, 32)
        self.F_AsyCA_2 = FAsyCA(32, 64)
        self.F_AsyCA_3 = FAsyCA(64, 128)
        self.F_AsyCA_4 = FAsyCA(128, 256)

        self.Block1 = MSA(256, 256)
        self.Block2 = MSA(256, 256)
        # 上采样
        self.trans_up_4 = Trans_Up(256, 128)
        self.trans_up_3 = Trans_Up(128, 64)
        self.trans_up_2 = Trans_Up(64, 32)
        self.trans_up_1 = Trans_Up(32, 16)
        # 融合
        self.up_4_fusion = nn.Sequential(
            nn.Conv2d(512, 256, 1, 1, 0),
            nn.InstanceNorm2d(256, affine=True),
            nn.GELU(),
        )
        self.up_3_fusion = nn.Sequential(
            nn.Conv2d(256, 128, 1, 1, 0),
            nn.InstanceNorm2d(128, affine=True),
            nn.GELU(),
        )
        self.up_2_fusion = nn.Sequential(
            nn.Conv2d(128, 64, 1, 1, 0),
            nn.InstanceNorm2d(64, affine=True),
            nn.GELU(),
        )
        self.up_1_fusion = nn.Sequential(
            nn.Conv2d(64, 32, 1, 1, 0),
            nn.InstanceNorm2d(32, affine=True),
            nn.GELU(),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(16, 16, 1, 1, 0),
            nn.InstanceNorm2d(16, affine=True),
            nn.GELU(),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.InstanceNorm2d(16, affine=True),
            nn.GELU(),
        )
        self.fusion2 = nn.Sequential(
            nn.Conv2d(16, 3, 3, 1, 1),
            nn.Tanh(),
        )

        self._compent4 = nn.Sequential(
            *[Color_Comp(128) for _ in range(2)]
        )
        self._compent3 = nn.Sequential(
            *[Color_Comp(64) for _ in range(2)]
        )
        self._compent2 = nn.Sequential(
            *[Color_Comp(32) for _ in range(2)]
        )
        self._compent1 = nn.Sequential(
            *[Color_Comp(16) for _ in range(2)]
        )

        self.main_com4 = nn.Sequential(
            *[Color_Comp(128) for _ in range(4)]
        )
        self.main_com3 = nn.Sequential(
            *[Color_Comp(64) for _ in range(4)]
        )
        self.main_com2 = nn.Sequential(
            *[Color_Comp(32) for _ in range(4)]
        )
        self.main_com1 = nn.Sequential(
            *[Color_Comp(16) for _ in range(4)]
        )

        self.fu_4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(128),
            nn.GELU()
        )
        self.fu_3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(64),
            nn.GELU()
        )
        self.fu_2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(32),
            nn.GELU()
        )
        self.fu_1 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(16),
            nn.GELU()
        )


    def forward(self, x, y):

        [down_51, down_41, down_31, down_21, down_11] = x
        [down_5, down_4, down_3, down_2, down_1] = y

        up_5 = self.Block1(down_51)   # 256,16,16

        Fa_4 = self.F_AsyCA_4(down_41)  # 256,16,16
        Fa_fusion_4 = self.up_4_fusion(torch.cat([up_5, Fa_4], dim=1))  # 256,16,16
        up_4 = self.trans_up_4(Fa_fusion_4)  # 128,32,32

        up_4 = self._compent4(down_4) + up_4
        up_4 = self.fu_4(self.main_com4(up_4))

        Fa_3 = self.F_AsyCA_3(down_31)  # 128,32,32
        Fa_fusion_3 = self.up_3_fusion(torch.cat([up_4, Fa_3], dim=1))  # 128,32,32
        up_3 = self.trans_up_3(Fa_fusion_3)  # 64 64 64

        up_3 = self._compent3(down_3) + up_3
        up_3 = self.fu_3(self.main_com3(up_3))

        Fa_2 = self.F_AsyCA_2(down_21)  # 64 64 64
        Fa_fusion_2 = self.up_2_fusion(torch.cat([up_3, Fa_2], dim=1))  # 64 64 64
        up_2 = self.trans_up_2(Fa_fusion_2)  # 32 128 128

        up_2 = self._compent2(down_2) + up_2
        up_2 = self.fu_2(self.main_com2(up_2))

        Fa_1 = self.F_AsyCA_1(down_11)  # 32 128 128
        Fa_fusion_1 = self.up_1_fusion(torch.cat([up_2, Fa_1], dim=1))  # 32 128 128
        up_1 = self.trans_up_1(Fa_fusion_1)  # 16 256 256

        up_1 = self._compent1(down_1) + up_1
        up_1 = self.fu_1(self.main_com1(up_1))

        feature = self.fusion(up_1)  # 1, 16, 256, 256
        outputs = self.fusion2(feature)

        return outputs

class FCCNet(nn.Module):

    def __init__(self):
        super(DFMUNet, self).__init__()
        self.EA = Encoder_A()
        self.EU = Encoder_U()
        self.GA = Generator_A()


    def forward(self, x):
       E_x = self.EA(x)
       y = 1-x
       E_u = self.EU(y)
       outp = self.GA(E_x, E_u)
       return outp

