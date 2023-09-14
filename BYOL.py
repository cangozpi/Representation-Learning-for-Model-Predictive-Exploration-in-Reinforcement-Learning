import torch
from torchvision import transforms as T
import torchvision.transforms.functional as TF
import random
import copy
from torch import nn
import torch.nn.functional as F


class Augment:
    """
    A stochastic data augmentation module
    Transforms any given data example randomly
    resulting in two correlated views of the same example,
    denoted x ̃i and x ̃j, which we consider as a positive pair.
    """
    def __init__(self, img_size=224, s=1):
        color_jitter = T.ColorJitter(
            0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s
        )
        blur = T.GaussianBlur((3, 3), (0.1, 2.0))

        self.train_transform = T.Compose([
            # T.ToTensor(),
            T.RandomResizedCrop(size=img_size),
            T.RandomHorizontalFlip(p=0.5),  # with 0.5 probability
            T.RandomApply([color_jitter], p=0.8),
            T.RandomApply([blur], p=0.5),
            # T.RandomGrayscale(p=0.2),
            # imagenet stats
            # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])            
        ])

    def __call__(self, x):
        return self.train_transform(x), self.train_transform(x)



class MLP(nn.Module):
    def __init__(self, dim, embedding_size=256, hidden_size=2048, batch_norm_mlp=False):
        super().__init__()
        norm = nn.BatchNorm1d(hidden_size) if batch_norm_mlp else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_size),
            norm,
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, embedding_size)
        )

    def forward(self, x):
        return self.net(x)


class AddProjHead(nn.Module):
    def __init__(self, model, in_features, layer_name, hidden_size=4096,
                 embedding_size=256, batch_norm_mlp=True):
        super(AddProjHead, self).__init__()
        self.backbone = model
        # remove last layer 'fc' or 'classifier'
        # setattr(self.backbone, layer_name, nn.Identity())
        # self.backbone.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # self.backbone.maxpool = torch.nn.Identity()
        # add mlp projection head
        self.projection = MLP(in_features, embedding_size, hidden_size=hidden_size, batch_norm_mlp=batch_norm_mlp)

    def forward(self, x, return_embedding=False):
        embedding = self.backbone(x)
        if return_embedding:
            return embedding
        return self.projection(embedding)


def loss_fn(x, y):
    # L2 normalization
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)


class EMA():
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.alpha + (1 - self.alpha) * new


class BYOL(nn.Module):
    def __init__(
            self,
            net,
            batch_norm_mlp=True,
            layer_name='fc',
            in_features=512,
            projection_size=256,
            projection_hidden_size=2048,
            moving_average_decay=0.99,
            use_momentum=True,
            use_cuda=False):
        """
        Args:
            net: model to be trained (i.e. backbone, feature extractor, f_theta)
            batch_norm_mlp: whether to use batchnorm1d in the mlp predictor and projector
            in_features: the number features that are produced by the backbone net i.e. resnet
            projection_size: the size of the output vector of the two identical MLPs
            projection_hidden_size: the size of the hidden vector of the two identical MLPs
            augment_fn2: apply different augmentation the second view
            moving_average_decay: t hyperparameter to control the influence in the target network weight update
            use_momentum: whether to update the target network
        """
        super().__init__()
        self.net = net
        self.online_model = AddProjHead(model=net, in_features=in_features,
                                         layer_name=layer_name,
                                         embedding_size=projection_size,
                                         hidden_size=projection_hidden_size,
                                         batch_norm_mlp=batch_norm_mlp)
        self.use_momentum = use_momentum
        self.target_model = self._get_target()
        self.target_ema_updater = EMA(moving_average_decay)
        self.online_predictor = MLP(projection_size, projection_size, projection_hidden_size)

        self.device = torch.device('cuda' if use_cuda else 'cpu')
        self.online_model = self.online_model.to(self.device)
        self.target_model = self.target_model.to(self.device)
        self.online_predictor = self.online_predictor.to(self.device)
    
    @torch.no_grad()
    def _get_target(self):
        return copy.deepcopy(self.online_model)
    
    @torch.no_grad()
    def update_moving_average(self):
        assert self.use_momentum, 'you do not need to update the moving average, since you have turned off momentum ' \
                                  'for the target encoder '
        assert self.target_model is not None, 'target encoder has not been created yet'

        for online_params, target_params in zip(self.online_model.parameters(), self.target_model.parameters()):
          old_weight, up_weight = target_params.data, online_params.data
          target_params.data = self.target_ema_updater.update_average(old_weight, up_weight)

    def forward(
            self,
            image_one, image_two=None,
            return_embedding=False):
        if return_embedding or (image_two is None):
            return self.online_model(image_one, return_embedding=True)

        # online projections: backbone + MLP projection
        online_proj_one = self.online_model(image_one)
        online_proj_two = self.online_model(image_two)

        # additional online's MLP head called predictor
        online_pred_one = self.online_predictor(online_proj_one)
        online_pred_two = self.online_predictor(online_proj_two)

        with torch.no_grad():
            # target processes the images and makes projections: backbone + MLP
            target_proj_one = self.target_model(image_one).detach_()
            target_proj_two = self.target_model(image_two).detach_()
            
        loss_one = loss_fn(online_pred_one, target_proj_two)
        loss_two = loss_fn(online_pred_two, target_proj_one)

        return (loss_one + loss_two).mean()


"""# Pre-train function"""
def training_step(model, data):
    (view1, view2), _ = data
    loss = model(view1.cuda(), view2.cuda())
    return loss

def train_one_epoch(model, train_dataloader, optimizer):
    model.train()
    total_loss = 0.
    num_batches = len(train_dataloader)
    for data in train_dataloader:
        optimizer.zero_grad()
        loss = training_step(model, data)
        loss.backward()
        optimizer.step()
        # EMA update
        model.update_moving_average()

        total_loss += loss.item()
        
    
    return total_loss/num_batches