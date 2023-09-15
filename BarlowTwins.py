# Adapted from https://github.com/facebookresearch/barlowtwins

from torch import nn
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image, ImageOps, ImageFilter
import random


class BarlowTwins(nn.Module):
    def __init__(self, backbone, in_features=512, projection_sizes=[8192, 8192, 8192], lambd=0.5, use_cuda=False):
        super().__init__()
        self.backbone = backbone
        self.lambd = lambd

        # projector
        sizes = [in_features] + projection_sizes
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)

        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)

        self.device = torch.device('cuda' if use_cuda else 'cpu')
        self.backbone = self.backbone.to(self.device)
        self.projector = self.projector.to(self.device)

    def off_diagonal(self, x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, y1, y2):
        z1 = self.projector(self.backbone(y1))
        z2 = self.projector(self.backbone(y2))

        # empirical cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2)

        # sum the cross-correlation matrix between all gpus
        c.div_(z1.shape[0])
        # torch.distributed.all_reduce(c)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = self.off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.lambd * off_diag
        return loss


class GaussianBlur(object): # from facebookresearch code
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img

class Solarization(object): # from facebookresearch code
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


class Transform:
    def __init__(self, img_size=224):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(img_size, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            # transforms.RandomGrayscale(p=0.2),

            # Facebook's GaussianBlur method cannot process batches of img tensors hence swapped it with transforms.* equivalent
            # GaussianBlur(p=1.0),
            # transforms.GaussianBlur(kernel_size=3),

            # Facebook's RandomSolarize method cannot process batches of img tensors hence swapped it with transforms.* equivalent
            # Solarization(p=0.0),
            # transforms.RandomSolarize(threshold=128, p=0.0),

            # transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                      std=[0.229, 0.224, 0.225])
        ])
        self.transform_prime = transforms.Compose([
            transforms.RandomResizedCrop(img_size, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            # transforms.RandomGrayscale(p=0.2),

            # Facebook's GaussianBlur method cannot process batches of img tensors hence swapped it with transforms.* equivalent
            # GaussianBlur(p=0.1),
            # transforms.GaussianBlur(kernel_size=3),

            # Facebook's RandomSolarize method cannot process batches of img tensors hence swapped it with transforms.* equivalent
            # Solarization(p=0.2),
            # transforms.RandomSolarize(threshold=128, p=0.2),

            # transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                      std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, x):
        y1 = self.transform(x)
        y2 = self.transform_prime(x)
        return y1, y2

