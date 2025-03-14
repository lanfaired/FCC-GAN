import torch
from torch import nn
from torch.nn import L1Loss
from torchvision import models, transforms

class GANLoss(nn.Module):
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


class PerceptionLoss_vgg19(nn.Module):
    def __init__(self):
        super(PerceptionLoss_vgg19, self).__init__()
        vgg = models.vgg19(pretrained=False)
        # conv2_2:7  conv3_2:12  conv4_2:21  conv5_2:30
        vgg.load_state_dict(torch.load('/root/autodl-tmp/data/vgg19-dcbb9e9d.pth'))
        loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.SmoothL1Loss()
        self.Norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def forward(self, out_images, target_images):
        out_images = self.Norm(out_images)
        target_images = self.Norm(target_images)
        perception_loss = self.mse_loss(self.loss_network(out_images), self.loss_network(target_images))
        return perception_loss


class FFT_LOSS(nn.Module):
    def __init__(self):
        super(FFT_LOSS, self).__init__()
        self.loss = L1Loss()

    def forward(self, out, label):
        fft_x = torch.fft.rfft2(out, norm='backward')
        mag_x = torch.abs(fft_x)
        pha_x = torch.angle(fft_x)

        fft = torch.fft.rfft2(label, norm='backward')
        mag_y = torch.abs(fft)
        pha_y = torch.angle(fft)

        mag_loss = self.loss(mag_x, mag_y)
        pha_loss = self.loss(pha_x, pha_y)

        return (mag_loss + pha_loss) * 0.5