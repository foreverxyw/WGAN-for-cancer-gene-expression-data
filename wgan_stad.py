
import argparse
import os
import matplotlib.pyplot as plt
import itertools
import pickle
# import imageio
import numpy as np
import math
import sys
import pandas as pd

import torchvision.transforms as transforms
from torchvision.utils import save_image
# import torch.utils.data as Data

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

from sklearn import preprocessing
from sklearn.model_selection import train_test_split 
from sklearn.svm import SVC 
from sklearn import metrics

# os.makedirs('images', exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=2000, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=16, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.00005, help='learning rate')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
parser.add_argument('--feature_size', type=int, default=801, help='size of each image dimension')
parser.add_argument('--channels', type=int, default=1, help='number of image channels')
parser.add_argument('--n_critic', type=int, default=5, help='number of training steps for discriminator per iter')
parser.add_argument('--clip_value', type=float, default=0.01, help='lower and upper clip value for disc. weights')
parser.add_argument('--sample_interval', type=int, default=400, help='interval betwen image samples')
opt = parser.parse_args()
print(opt)

img_shape = (opt.feature_size, )

cuda = True if torch.cuda.is_available() else False

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [  nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, normalize=False),
            *block(128, 256, normalize=False),
            *block(256, 512, normalize=False),
            *block(512, 1024, normalize=False),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *img_shape)
        return img

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1)
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity


def show_train_hist(hist, show = False, save = False, path = 'Train_hist.eps'):
    x = range(len(hist['D_losses']))

    y1 = hist['D_losses']
    y2 = hist['G_losses']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

x = Tensor(normal_train)   # x data (torch tensor)
y = torch.from_numpy(label_normal_train)     # y data (torch tensor)

print(x.shape, y.shape)
# 先转换成 torch 能识别的 Dataset
torch_dataset = torch.utils.data.TensorDataset(x, y)

dataloader = DataLoader(dataset=torch_dataset,
                             batch_size=opt.batch_size, 
                             shuffle=True,)

# Optimizers
optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=opt.lr)
optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=opt.lr)


# make path to store result
os.makedirs('results', exist_ok=True)
root = 'results/gene_data_normal_0323/'
os.makedirs(root, exist_ok=True)
root_model = root + 'model/' 
os.makedirs(root_model, exist_ok=True)
root_data = root + 'data/'
os.makedirs(root_data, exist_ok=True)

# ----------
#  Training
# ----------

batches_done = 0
train_hist = {}
train_hist['D_losses'] = []
train_hist['G_losses'] = []
for epoch in range(opt.n_epochs):
    D_losses = []
    G_losses = []
    for i, (imgs, _) in enumerate(dataloader):

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # ---------------------
        #  Train Discriminator
        # ---------------------
        for j in range(opt.n_critic):
            optimizer_D.zero_grad()

            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

            # Generate a batch of images
            fake_imgs = generator(z).detach()
            # Adversarial loss
            loss_D = -torch.mean(discriminator(real_imgs)) + torch.mean(discriminator(fake_imgs))

            loss_D.backward()
            optimizer_D.step()

            # Clip weights of discriminator
            for p in discriminator.parameters():
                p.data.clamp_(-opt.clip_value, opt.clip_value)

            D_losses.append(loss_D.item())

        # Train the generator every n_critic iterations
        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Generate a batch of images
        gen_imgs = generator(z)
        # Adversarial loss
        loss_G = -torch.mean(discriminator(gen_imgs))

        loss_G.backward()
        optimizer_G.step()

        G_losses.append(loss_G.item())
        print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch+1, opt.n_epochs,
                                                            batches_done % len(dataloader), len(dataloader),
                                                            torch.mean(torch.FloatTensor(D_losses)), torch.mean(torch.FloatTensor(G_losses))))

        batches_done += 1
    train_hist['D_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
    train_hist['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))

print("Training finish!... save training results")
torch.save(generator.state_dict(), root_model+"generator_param_205.pkl")
torch.save(discriminator.state_dict(), root_model+"discriminator_param_205.pkl")
with open(root_model+"train_hist_205.pkl", 'wb') as f:
    pickle.dump(train_hist, f)

show_train_hist(train_hist, save=True, path=root+'train_hist_205.eps')

# Sample noise as generator input
z = Tensor(np.random.normal(0, 1, (205, opt.latent_dim)))
gen_imgs = generator(z)
gen_imgs = pd.DataFrame(gen_imgs.detach().numpy())  #NEW
gen_imgs.to_csv(root_data+'normal_205.csv', index=False, header=False)