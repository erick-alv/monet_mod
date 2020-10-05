# License: MIT
# Author: Karl Stelzner

import torch
import torch.nn as nn
import torch.distributions as dists
import numpy as np

import torchvision


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

class UNet(nn.Module):
    def __init__(self, num_blocks, in_channels, out_channels, channel_base=64):
        super().__init__()
        self.num_blocks = num_blocks
        self.down_convs = nn.ModuleList()
        cur_in_channels = in_channels
        for i in range(num_blocks):
            self.down_convs.append(double_conv(cur_in_channels,
                                               channel_base * 2**i))
            cur_in_channels = channel_base * 2**i

        self.tconvs = nn.ModuleList()
        for i in range(num_blocks-1, 0, -1):
            self.tconvs.append(nn.ConvTranspose2d(channel_base * 2**i,
                                                  channel_base * 2**(i-1),
                                                  2, stride=2))

        self.up_convs = nn.ModuleList()
        for i in range(num_blocks-2, -1, -1):
            self.up_convs.append(double_conv(channel_base * 2**(i+1), channel_base * 2**i))

        self.final_conv = nn.Conv2d(channel_base, out_channels, 1)

    def forward(self, x):
        intermediates = []
        cur = x
        for down_conv in self.down_convs[:-1]:
            cur = down_conv(cur)
            intermediates.append(cur)
            cur = nn.MaxPool2d(2)(cur)

        cur = self.down_convs[-1](cur)

        for i in range(self.num_blocks-1):
            cur = self.tconvs[i](cur)
            cur = torch.cat((cur, intermediates[-i -1]), 1)
            cur = self.up_convs[i](cur)

        return self.final_conv(cur)

class AttentionNet(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        self.unet = UNet(num_blocks=conf.num_blocks,
                         in_channels=4,
                         out_channels=2,
                         channel_base=conf.channel_base)

    def forward(self, x, scope):
        inp = torch.cat((x, scope), 1)
        logits = self.unet(inp)
        alpha = torch.softmax(logits, 1)
        # output channel 0 represents alpha_k,
        # channel 1 represents (1 - alpha_k).
        mask = scope * alpha[:, 0:1]
        new_scope = scope * alpha[:, 1:2]
        return mask, new_scope

class EncoderNet(nn.Module):
    def __init__(self, width, height, device, latent_size=16, full_connected_size=256, input_channels=4,
                 kernel_size=3, encoder_stride=2, conv_size1=32, conv_size2=64):
        super().__init__()
        self.device = device
        self.conv_size1 = conv_size1
        self.conv_size2 = conv_size2
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=conv_size1,
                      kernel_size=kernel_size, stride=encoder_stride),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=conv_size1, out_channels=conv_size1,
            kernel_size=kernel_size, stride=encoder_stride),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=conv_size1, out_channels=conv_size2,
            kernel_size=kernel_size, stride=encoder_stride),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=conv_size2, out_channels=conv_size2,
            kernel_size=kernel_size, stride=encoder_stride),
            nn.ReLU(inplace=True)
        )

        red_width = width
        red_height = height
        #todo is 4 since for conv layers; if less the have to reduce this as well
        for i in range(4):
            red_width = (red_width - 1) // 2
            red_height = (red_height - 1) // 2

        self.red_width = red_width
        self.red_height = red_height

        self.fc1 = nn.Sequential(
            nn.Linear(conv_size2 * red_width * red_height, full_connected_size),
            nn.ReLU(inplace=True),
        )
        self.fc21 = nn.Linear(full_connected_size, latent_size)
        self.fc22 = nn.Linear(full_connected_size, latent_size)

    def forward(self, x):
        cx = self.convs(x)
        f_cx = cx.reshape(-1, self.red_width * self.red_height * self.conv_size2)
        #x = x.view(x.shape[0], -1)
        e = self.fc1(f_cx)
        return self.fc21(e), self.fc22(e)

class DecoderNet(nn.Module):
    def __init__(self, width, height, device, latent_size, output_channels=4,
                 kernel_size=3, conv_size1=32, decoder_stride=1):
        super().__init__()
        self.device = device
        self.height = height
        self.width = width
        self.latent_size = latent_size
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=latent_size+2, out_channels=conv_size1,
                      kernel_size=kernel_size, stride=decoder_stride),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=conv_size1, out_channels=conv_size1,
                      kernel_size=kernel_size, stride=decoder_stride),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=conv_size1, out_channels=conv_size1,
                      kernel_size=kernel_size, stride=decoder_stride),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=conv_size1, out_channels=conv_size1,
                      kernel_size=kernel_size, stride=decoder_stride),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=conv_size1, out_channels=output_channels, kernel_size=1),
        )
        ys = torch.linspace(-1, 1, self.height + 8)
        xs = torch.linspace(-1, 1, self.width + 8)
        ys, xs = torch.meshgrid(ys, xs)
        coord_map = torch.stack((ys, xs)).unsqueeze(0)
        self.register_buffer('coord_map_const', coord_map)

    def forward(self, z):
        z_tiled = z.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.height + 8, self.width + 8)
        coord_map = self.coord_map_const.repeat(z.shape[0], 1, 1, 1)
        inp = torch.cat((z_tiled, coord_map), 1)
        result = self.convs(inp)
        return result


class Monet(nn.Module):
    def __init__(self, conf, height, width, device, latent_size, bg_sigma=0.09,fg_sigma=0.11,):
        super().__init__()
        self.conf = conf
        self.attention = AttentionNet(conf)
        self.latent_size = latent_size
        self.bg_sigma=bg_sigma
        self.fg_sigma=fg_sigma
        self.height = height
        self.width = width
        self.color_channels = 3
        self.beta = 0.5
        self.gamma = 0.25

        self.encoder = EncoderNet(device = device, latent_size=latent_size, height=height, width=width)
        self.decoder = DecoderNet(width=width, height=height, device=device, latent_size=latent_size)




    def forward_old(self, x):
        scope = torch.ones_like(x[:, 0:1])
        masks = []
        for i in range(self.conf.num_slots-1):
            mask, scope = self.attention(x, scope)
            masks.append(mask)
        masks.append(scope)
        loss = torch.zeros_like(x[:, 0, 0, 0])
        mask_preds = []
        full_reconstruction = torch.zeros_like(x)
        p_xs = torch.zeros_like(loss)
        kl_zs = torch.zeros_like(loss)
        for i, mask in enumerate(masks):
            z, kl_z = self.__encoder_step(x, mask)
            sigma = self.conf.bg_sigma if i == 0 else self.conf.fg_sigma
            p_x, x_recon, mask_pred = self.__decoder_step(x, z, mask, sigma)
            mask_preds.append(mask_pred)
            loss += -p_x + self.beta * kl_z
            p_xs += -p_x
            kl_zs += kl_z
            full_reconstruction += mask * x_recon

        masks = torch.cat(masks, 1)
        tr_masks = torch.transpose(masks, 1, 3)
        q_masks = dists.Categorical(probs=tr_masks)
        stacked_preds = torch.stack(mask_preds, 3)
        q_masks_recon = dists.Categorical(logits=stacked_preds)
        kl_masks = dists.kl_divergence(q_masks, q_masks_recon)
        kl_masks = torch.sum(kl_masks, [1, 2])
        # print('px', p_xs.mean().item(),
        #       'kl_z', kl_zs.mean().item(),
        #       'kl masks', kl_masks.mean().item())
        loss += self.gamma * kl_masks
        return {'loss': loss,
                'masks': masks,
                'reconstructions': full_reconstruction,
                'p_xs':p_xs,
                'kl_zs':kl_zs}


    def __encoder_step(self, x, mask):
        encoder_input = torch.cat((x, mask), 1)
        means, sigmas = self.encoder(encoder_input)
        # THIS belonged to original code I suppose it was to enforce a normal distribution
        means = torch.sigmoid(means) * 6 - 3
        sigmas = torch.sigmoid(sigmas) * 3
        dist = dists.Normal(means, sigmas)
        dist_0 = dists.Normal(0., sigmas)
        z = means + dist_0.sample()
        q_z = dist.log_prob(z)
        kl_z = dists.kl_divergence(dist, dists.Normal(0., 1.))
        kl_z = torch.sum(kl_z, 1)
        return z, kl_z

    def __encoder_step_own(self, x, mask):
        encoder_input = torch.cat((x, mask), 1)
        mu, logvar = self.encoder(encoder_input)
        return mu, logvar

    def encode(self, x):
        scope = torch.ones_like(x[:, 0:1])
        masks = []
        for i in range(self.conf.num_slots-1):
            mask, scope = self.attention(x, scope)
            masks.append(mask)
        masks.append(scope)
        mu_s = []
        logvar_s = []
        for i, mask in enumerate(masks):
            mu, logvar = self.__encoder_step_own(x, mask)
            mu_s.append(mu)
            logvar_s.append(logvar)

        return mu_s, logvar_s, masks



    def forward(self, x):
        mu_s, logvar_s, masks = self.encode(x)
        z_s = [self.__reparameterize(mu_s[i], logvar_s[i]) for i in range(len(mu_s))]
        full_reconstruction, x_recon_s, mask_pred_s = self.decode(z_s, masks)
        return mu_s, logvar_s, masks, full_reconstruction, x_recon_s, mask_pred_s

    def __reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return (mu + eps * std)

    def __decoder_step(self, x, z, mask, sigma):
        decoder_output = self.decoder(z)
        x_recon = torch.sigmoid(decoder_output[:, :3])
        mask_pred = decoder_output[:, 3]
        dist = dists.Normal(x_recon, sigma)
        p_x = dist.log_prob(x)
        p_x *= mask
        p_x = torch.sum(p_x, [1, 2, 3])
        return p_x, x_recon, mask_pred

    def __decoder_step_own(self, z):
        decoder_output = self.decoder(z)
        x_recon = torch.sigmoid(decoder_output[:, :3])
        mask_pred = decoder_output[:, 3]
        return x_recon, mask_pred

    def decode(self, z_s, masks):
        full_reconstruction = torch.zeros((masks[0].shape[0], self.color_channels, self.width, self.height)).cuda()
        x_recon_s, mask_pred_s = [], []
        for i in range(len(masks)):
            x_recon, mask_pred = self.__decoder_step_own(z_s[i])
            x_recon_s.append(x_recon)
            mask_pred_s.append(mask_pred)
            full_reconstruction += x_recon*masks[i]

        return full_reconstruction, x_recon_s, mask_pred_s


def train(epoch, model, optimizer, device, log_interval, train_file, batch_size, beta, gamma, bg_sigma, fg_sigma):
    model.train()
    train_loss = 0
    data_set = np.load(train_file)
    data_set = data_set[:10000]

    data_size = len(data_set)
    data_set = np.split(data_set, data_size / batch_size)

    for batch_idx, data in enumerate(data_set):
        data = torch.from_numpy(data).float().to(device)
        #data /= 255#todo this must be used once trained with own images
        data = data.permute([0, 3, 1, 2])
        optimizer.zero_grad()
        mu_s, logvar_s, masks, full_reconstruction, x_recon_s, mask_pred_s = model(data)
        loss_batch = loss_function(data, x_recon_s, masks, mask_pred_s, mu_s, logvar_s, beta, gamma, bg_sigma, fg_sigma)
        loss = torch.mean(loss_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx + 1) * len(data), data_size,
                       100. * (batch_idx + 1) / len(data_set),
                       loss.item() / len(data)))
            print('Loss: ', loss.item() / len(data))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / data_size))
    return numpify(data),numpify(torch.cat(masks, dim=1)),numpify(full_reconstruction)

def loss_function(x, x_recon_s, masks, mask_pred_s, mu_s, logvar_s, beta, gamma, bg_sigma, fg_sigma):
    batch_size = x.shape[0]
    p_xs = torch.zeros(batch_size).cuda()
    kl_z = torch.zeros(batch_size).cuda()
    for i in range(len(masks)):
        kld = -0.5 * torch.sum(1 + logvar_s[i] - mu_s[i].pow(2) - logvar_s[i].exp(), dim=1)
        kl_z += kld
        if i == 0:
            sigma = bg_sigma
        else:
            sigma = fg_sigma
        dist = dists.Normal(x_recon_s[i], sigma)
        p_x = dist.log_prob(x)
        p_x *= masks[i]
        p_x = torch.sum(p_x, [1, 2, 3])
        p_xs += -p_x

    masks = torch.cat(masks, 1)
    tr_masks = torch.transpose(masks, 1, 3)
    q_masks = dists.Categorical(probs=tr_masks)
    stacked_mask_preds = torch.stack(mask_pred_s, 3)
    q_masks_recon = dists.Categorical(logits=stacked_mask_preds)
    kl_masks = dists.kl_divergence(q_masks, q_masks_recon)
    kl_masks = torch.sum(kl_masks, [1, 2])
    loss = gamma * kl_masks + p_xs + beta* kl_z
    return loss

def numpify(tensor):
    return tensor.cpu().detach().numpy()

def print_image_stats(images, name):
    print(name, '0 min/max', images[:, 0].min().item(), images[:, 0].max().item())
    print(name, '1 min/max', images[:, 1].min().item(), images[:, 1].max().item())
    print(name, '2 min/max', images[:, 2].min().item(), images[:, 2].max().item())


