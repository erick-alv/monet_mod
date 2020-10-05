# License: MIT
# Author: Karl Stelzner

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np
#import visdom

import os

import model
import datasets
import config
from PIL import Image

def visualize_masks(epoch, imgs, masks, recons):
    # print('recons min/max', recons[:, 0].min().item(), recons[:, 0].max().item())
    # print('recons1 min/max', recons[:, 1].min().item(), recons[:, 1].max().item())
    # print('recons2 min/max', recons[:, 2].min().item(), recons[:, 2].max().item())
    recons = np.clip(recons, 0., 1.)
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0)]
    colors.extend([(c[0]//2, c[1]//2, c[2]//2) for c in colors])
    colors.extend([(c[0]//4, c[1]//4, c[2]//4) for c in colors])
    seg_maps = np.zeros_like(imgs)
    masks = np.argmax(masks, 1)
    for i in range(imgs.shape[0]):
        for y in range(imgs.shape[2]):
            for x in range(imgs.shape[3]):
                seg_maps[i, :, y, x] = colors[masks[i, y, x]]

    imgs *= 255.0
    recons *= 255.0

    imgs = np.transpose(imgs, (0, 2, 3, 1))
    imgs = np.concatenate(imgs, axis=1)
    seg_maps = np.transpose(seg_maps, (0, 2, 3, 1))
    seg_maps = np.concatenate(seg_maps, axis=1)
    recons = np.transpose(recons, (0, 2, 3, 1))
    recons = np.concatenate(recons, axis=1)
    all_im_array = np.concatenate((imgs, seg_maps, recons), axis=0)
    all_im = Image.fromarray(all_im_array.astype(np.uint8))
    all_im.save('./results/recons_{}.png'.format(epoch))

def run_training(monet, conf, train_file, device):
    if conf.load_parameters and os.path.isfile(conf.checkpoint_file):
        monet.load_state_dict(torch.load(conf.checkpoint_file))
        print('Restored parameters from', conf.checkpoint_file)
    else:
        for w in monet.parameters():
            std_init = 0.01
            nn.init.normal_(w, mean=0., std=std_init)
        print('Initialized parameters')

    optimizer = optim.RMSprop(monet.parameters(), lr=1e-4)

    for epoch in range(conf.num_epochs):
        imgs, masks, recons = model.train(epoch=epoch, model=monet, optimizer=optimizer, device=device, log_interval=200,
                                          train_file=train_file, batch_size=conf.batch_size, beta=conf.beta,
                                          gamma=conf.gamma, bg_sigma=conf.bg_sigma, fg_sigma=conf.fg_sigma)
        visualize_masks(epoch, imgs, masks, recons)

    torch.save(monet.state_dict(), conf.checkpoint_file)
    print('training done')


def sprite_experiment():
    conf = config.sprite_config
    #transform = transforms.Compose([transforms.ToTensor(),
    #                                transforms.Lambda(lambda x: x.float()),
    #                                ])
    #trainset = datasets.Sprites(conf.data_dir, train=True, transform=transform)
    #trainloader = torch.utils.data.DataLoader(trainset,
    #                                          batch_size=conf.batch_size,
    #                                          shuffle=True, num_workers=2)
    monet = model.Monet(conf, device=None, latent_size=16, height=64, width=64).cuda()
    if conf.parallel:
        monet = nn.DataParallel(monet)
    device = torch.device("cuda")
    run_training(monet, conf, train_file='./data/sprites_25000_64.npy', device=device)

def clevr_experiment():
    conf = config.clevr_config
    # Crop as described in appendix C
    crop_tf = transforms.Lambda(lambda x: transforms.functional.crop(x, 29, 64, 192, 192))
    drop_alpha_tf = transforms.Lambda(lambda x: x[:3])
    transform = transforms.Compose([crop_tf,
                                    transforms.Resize((128, 128)),
                                    transforms.ToTensor(),
                                    drop_alpha_tf,
                                    transforms.Lambda(lambda x: x.float()),
                                   ])
    trainset = datasets.Clevr(conf.data_dir,
                              transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=conf.batch_size,
                                              shuffle=True, num_workers=8)
    monet = model.Monet(conf, 128, 128).cuda()
    if conf.parallel:
        monet = nn.DataParallel(monet)
    run_training(monet, conf, trainloader)

if __name__ == '__main__':
    # clevr_experiment()
    sprite_experiment()

