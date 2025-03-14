import argparse, torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn as nn
from torchvision.utils import save_image
import time

from models.FCCNet import FCCNet
from mydatasets.datasets import SUIMDataset
from models.DFLNet import  NLayerDiscriminator
from util.image_pool import ImagePool
from lossdir.loss import GANLoss, PerceptionLoss_vgg19

import logging

# 设置日志配置
logging.basicConfig(filename='training_loss.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


parser = argparse.ArgumentParser()
parser.add_argument("--num_epochs", type=int, default=201, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=20, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")  # 0.0003
args = parser.parse_args()


fake_pool = ImagePool(50)
criterionGAN = GANLoss('lsgan').cuda()
loss_cont = PerceptionLoss_vgg19().cuda()
smooth_l1_loss = nn.SmoothL1Loss().cuda()
# loss_func = nn.CosineEmbeddingLoss(margin=0.5).cuda()

num_epochs = args.num_epochs
batch_size = args.batch_size
lr_rate = args.lr

network = FCCNet()
network = network.cuda()
discrimitor = NLayerDiscriminator()
discrimitor = discrimitor.cuda()

optimizer_g = torch.optim.AdamW(network.parameters(), lr=lr_rate, betas=(0.7,0.999))
scheduler_g = CosineAnnealingLR(optimizer=optimizer_g, T_max=50, eta_min=1e-5)

optimizer_d = torch.optim.AdamW(discrimitor.parameters(),lr=0.0001, betas=(0.7,0.999))
scheduler_d = CosineAnnealingLR(optimizer=optimizer_d, T_max=50, eta_min=1e-5)

save_img_num = 0
total_iters = 0
for epoch in range(num_epochs):
    dataloader = DataLoader(SUIMDataset(), batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    network.train()
    discrimitor.train()
    start_time = time.time()
    epoch_iter = 0
    for i, batch in enumerate(dataloader):
        iter_start_time = time.time()
        epoch_iter += batch_size
        total_iters += batch_size

        input = batch['input'].cuda()
        lab = batch['lab'].cuda()
        target = batch['target'].cuda()


        # 先训练判别器
        optimizer_d.zero_grad()
        fake_img = network(input, lab)
        # fake_img = fake_pool.query(output)

        pred_real = discrimitor(target)
        loss_D_real = criterionGAN(pred_real, True)
        pred_fake = discrimitor(fake_img.detach())
        loss_D_fake = criterionGAN(pred_fake, False)
        loss_D = (loss_D_real + loss_D_fake) * 0.5 * 5
        loss_D.backward()
        optimizer_d.step()

        # 训练生成器
        optimizer_g.zero_grad()
        output = network(input, lab)
        loss_gan = criterionGAN(discrimitor(output),True)
        loss_vgg = loss_cont(output,target)
        loss_l1 = smooth_l1_loss(output,target)
        # label_ang = torch.tensor([1])
        #loss_ang = loss_func(output, target, label_ang)
        loss_G = loss_gan + loss_vgg * 3 + loss_l1 * 7
        loss_G.backward()
        optimizer_g.step()

        # 输出信息
        if total_iters % 400 == 0:
            t_comp = time.time() - iter_start_time
            message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, epoch_iter, t_comp)
            message += '%s: %.3f ' % ("loss_gan", loss_gan.item())
            message += '%s: %.3f ' % ("loss_dis", loss_D.item())
            message += '%s: %.3f ' % ("loss_vgg", loss_vgg.item())
            message += '%s: %.3f ' % ("loss_l1", loss_l1.item())
            # message += '%s: %.3f ' % ("loss_ang", loss_ang.item())
            total_loss = loss_gan.item() + loss_vgg.item() + loss_l1.item()
            message += '%s: %.3f ' % ("total_loss", total_loss)
            logging.info(f'Epoch [{epoch + 1}/200] Num {total_iters}:loss_gan: {loss_gan.item():.3f},loss_dis: {loss_D.item():.3f},loss_vgg: {loss_vgg.item():.3f},loss_l1:{loss_l1.item():.3f},total_loss:{ total_loss:.3f}')
            print(message)
        #保存图片
        if total_iters % 10000 == 0:
            save_img_num += 1
            save_image(input.cpu().data,'./result/raw_{}.jpg'.format(save_img_num))
            save_image(target.cpu().data, './result/label_{}.jpg'.format(save_img_num))
            save_image(output.cpu().data, './result/my_{}.jpg'.format(save_img_num))
            print('---------save success---------')
        # 保存权重
        if total_iters % 20000 == 0:
            torch.save(network.state_dict(),'./pathweight/netwk_{}.pth'.format(save_img_num))
            torch.save(discrimitor.state_dict(), './pathweight/discrimitor_{}.pth'.format(save_img_num))
            print('---------save pth success--------')

    scheduler_d.step()
    scheduler_g.step()









