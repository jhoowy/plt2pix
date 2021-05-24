import os, gzip, torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import imageio
import matplotlib as mpl
mpl.use('Agg')  # or whatever other backend that you want
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

def save_images(images, size, path):
    image = np.squeeze(merge(images, size))
    if np.max(image) <= 1.0:
        image *= 255
    return imageio.imwrite(path, image.astype(np.uint8))

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3,4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3]==1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter ''must have dimensions: HxW or HxWx3 or HxWx4')

def loss_plot(hist, result_path, epoch):
    x = range(len(hist['D_loss']))

    y1 = hist['D_loss']
    y2 = hist['G_loss']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Iter')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    path = result_path / 'plt2pix_loss_{}.png'.format(epoch)

    plt.savefig(str(path))

    plt.close()

def plt_to_img(plt_tensor):
    b, n = plt_tensor.shape
    plt_tensor = plt_tensor.reshape(b, -1, 1, 3) * 255.
    plt_tensor = torch.transpose(plt_tensor, 1, 3)
    img = torch.nn.Upsample(scale_factor=20)(plt_tensor)

    return img