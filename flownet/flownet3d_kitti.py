import glob
import json
import matplotlib.pyplot as plt
from multiprocessing import Manager
import numpy as np
import os
import time
import random

import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import kaolin as kal
from kaolin.models.PointNet2 import furthest_point_sampling
from kaolin.models.PointNet2 import fps_gather_by_index
from kaolin.models.PointNet2 import ball_query
from kaolin.models.PointNet2 import group_gather_by_index
from kaolin.models.PointNet2 import three_nn
from kaolin.models.PointNet2 import three_interpolate

import tensorflow as tf

class SceneflowDataset(Dataset):
    def __init__(self, npoints=2048, root='data/data_processed_maxcut_35_20k_2k_8192', train=True, cache=None):
        self.npoints = npoints
        self.train = train
        self.root = root
        if self.train:
            self.datapath = glob.glob(os.path.join(self.root, 'TRAIN*.npz'))
        else:
            self.datapath = glob.glob(os.path.join(self.root, 'TEST*.npz'))
        
        if cache is None:
            self.cache = {}
        else:
            self.cache = cache
        
        self.cache_size = 30000

        ###### deal with one bad datapoint with nan value
        self.datapath = [d for d in self.datapath if 'TRAIN_C_0140_left_0006-0' not in d]
        ######

    def __getitem__(self, index):
        if index in self.cache:
            pos1, pos2, color1, color2, flow, mask1 = self.cache[index]
        else:
            fn = self.datapath[index]
            with open(fn, 'rb') as fp:
                data = np.load(fp)
                pos1 = data['points1'].astype('float32')
                pos2 = data['points2'].astype('float32')
                color1 = data['color1'].astype('float32') / 255
                color2 = data['color2'].astype('float32') / 255
                flow = data['flow'].astype('float32')
                mask1 = data['valid_mask1']

            if len(self.cache) < self.cache_size:
                self.cache[index] = (pos1, pos2, color1, color2, flow, mask1)

        if self.train:
            n1 = pos1.shape[0]
            sample_idx1 = np.random.choice(n1, self.npoints, replace=False)
            n2 = pos2.shape[0]
            sample_idx2 = np.random.choice(n2, self.npoints, replace=False)

            pos1 = pos1[sample_idx1, :]
            pos2 = pos2[sample_idx2, :]
            color1 = color1[sample_idx1, :]
            color2 = color2[sample_idx2, :]
            flow = flow[sample_idx1, :]
            mask1 = mask1[sample_idx1]
        else:
            pos1 = pos1[:self.npoints, :]
            pos2 = pos2[:self.npoints, :]
            color1 = color1[:self.npoints, :]
            color2 = color2[:self.npoints, :]
            flow = flow[:self.npoints, :]
            mask1 = mask1[:self.npoints]

        pos1_center = np.mean(pos1, 0)
        pos1 -= pos1_center
        pos2 -= pos1_center
        
        pos1 = torch.from_numpy(pos1).t()
        pos2 = torch.from_numpy(pos2).t()
        color1 = torch.from_numpy(color1).t()
        color2 = torch.from_numpy(color2).t()
        flow = torch.from_numpy(flow).t()
        mask1 = torch.from_numpy(mask1)

        return pos1, pos2, color1, color2, flow, mask1

    def __len__(self):
        return len(self.datapath)


class Kitti_SceneflowDataset(Dataset):
    def __init__(self, npoints=4096, root='data/kitti_rm_ground', train=True, cache=None):
        self.npoints = npoints
        self.train = train
        self.root = root
        if self.train:
            self.datapath = glob.glob(os.path.join(self.root+'/train', '*.npz'))
        else:
            self.datapath = glob.glob(os.path.join(self.root+'/test', '*.npz'))

        if cache is None:
            self.cache = {}
        else:
            self.cache = cache

        self.cache_size = 30000


    def __getitem__(self, index):
        if index in self.cache:
            pos1, pos2, flow = self.cache[index]
        else:
            fn = self.datapath[index]
            with open(fn, 'rb') as fp:
                data = np.load(fp)
                pos1 = data['pos1'].astype('float32')
                pos2 = data['pos2'].astype('float32')
                flow = data['gt'].astype('float32')
                #pos1 = pos1[:,[2,0,1]]
                #pos2 = pos2[:,[2,0,1]]
                #flow = flow[:,[2,0,1]]

            if len(self.cache) < self.cache_size:
                self.cache[index] = (pos1, pos2, flow)

        n1 = pos1.shape[0]
        if n1 >= self.npoints:
            sample_idx1 = np.random.choice(n1, self.npoints, replace=False)
        else:
            sample_idx1 = np.concatenate((np.arange(n1), np.random.choice(n1, self.npoints - n1, replace=True)),
                                         axis=-1)
        n2 = pos2.shape[0]
        if n2 > self.npoints:
            sample_idx2 = np.random.choice(n2, self.npoints, replace=False)
        else:
            sample_idx2 = np.concatenate((np.arange(n2), np.random.choice(n2, self.npoints - n2, replace=True)),
                                        axis=-1)

        pos1 = pos1[sample_idx1, :]
        pos2 = pos2[sample_idx2, :]
        flow = flow[sample_idx1, :]

        color1 = np.zeros([self.npoints, 3]).astype(np.float32)
        color2 = np.zeros([self.npoints, 3]).astype(np.float32)
        mask1 = np.ones([self.npoints]).astype(np.float32)

        pos1 = torch.from_numpy(pos1).t()
        pos2 = torch.from_numpy(pos2).t()
        color1 = torch.from_numpy(color1).t()
        color2 = torch.from_numpy(color2).t()
        flow = torch.from_numpy(flow).t()
        mask1 = torch.from_numpy(mask1)

        return pos1, pos2, color1, color2, flow, mask1

    def __len__(self):
        return len(self.datapath)

train_set = Kitti_SceneflowDataset(train=True)
points1, points2, color1, color2, flow, mask1 = train_set[0]

print(points1.shape, points1.dtype)
print(points2.shape, points2.dtype)
print(color1.shape, color1.dtype)
print(color2.shape, color2.dtype)
print(flow.shape, flow.dtype)
print(mask1.shape, mask1.dtype)

def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def pdist2squared(x, y):
    xx = (x**2).sum(dim=1).unsqueeze(2)
    yy = (y**2).sum(dim=1).unsqueeze(1)
    dist = xx + yy - 2.0 * torch.bmm(x.permute(0, 2, 1), y)
    dist[dist != dist] = 0
    dist = torch.clamp(dist, 0.0, np.inf)
    return dist

def parameter_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class ClippedStepLR(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, step_size, min_lr, gamma=0.1, last_epoch=-1):
        self.step_size = step_size
        self.min_lr = min_lr
        self.gamma = gamma
        super(ClippedStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [max(base_lr * self.gamma ** (self.last_epoch // self.step_size), self.min_lr)
                for base_lr in self.base_lrs]
    
def criterion(pred_flow, flow, mask):
    loss = torch.mean(mask * torch.sum((pred_flow - flow) * (pred_flow - flow), dim=1) / 2.0)
    return loss

def error(pred, labels, mask):
    pred = pred.permute(0,2,1).cpu().numpy()
    labels = labels.permute(0,2,1).cpu().numpy()
    mask = mask.cpu().numpy()
    
    err = np.sqrt(np.sum((pred - labels)**2, 2) + 1e-20)

    gtflow_len = np.sqrt(np.sum(labels*labels, 2) + 1e-20) # B,N
    acc050 = np.sum(np.logical_or((err <= 0.05)*mask, (err/gtflow_len <= 0.05)*mask), axis=1)
    acc010 = np.sum(np.logical_or((err <= 0.1)*mask, (err/gtflow_len <= 0.1)*mask), axis=1)

    mask_sum = np.sum(mask, 1)
    acc050 = acc050[mask_sum > 0] / mask_sum[mask_sum > 0]
    acc050 = np.mean(acc050)
    acc010 = acc010[mask_sum > 0] / mask_sum[mask_sum > 0]
    acc010 = np.mean(acc010)

    epe = np.sum(err * mask, 1)[mask_sum > 0] / mask_sum[mask_sum > 0]
    epe = np.mean(epe)
    return epe, acc050, acc010

class Sample(nn.Module):
    def __init__(self, num_points):
        super(Sample, self).__init__()
        
        self.num_points = num_points
        
    def forward(self, points):
        new_points_ind = furthest_point_sampling(points.permute(0, 2, 1).contiguous(), self.num_points)
        new_points = fps_gather_by_index(points, new_points_ind)
        return new_points
    
class Group(nn.Module):
    def __init__(self, radius, num_samples, knn=False):
        super(Group, self).__init__()
        
        self.radius = radius
        self.num_samples = num_samples
        self.knn = knn
        
    def forward(self, points, new_points, features):
        if self.knn:
            dist = pdist2squared(points, new_points)
            ind = dist.topk(self.num_samples, dim=1, largest=False)[1].int().permute(0, 2, 1).contiguous()
        else:
            ind = ball_query(self.radius, self.num_samples, points.permute(0, 2, 1).contiguous(),
                             new_points.permute(0, 2, 1).contiguous(), False)
        grouped_points = group_gather_by_index(points, ind)
        grouped_points -= new_points.unsqueeze(3)
        grouped_features = group_gather_by_index(features, ind)
        new_features = torch.cat([grouped_points, grouped_features], dim=1)
        return new_features

class SetConv(nn.Module):
    def __init__(self, num_points, radius, num_samples, in_channels, out_channels):
        super(SetConv, self).__init__()
        
        self.sample = Sample(num_points)
        self.group = Group(radius, num_samples)
        
        layers = []
        out_channels = [in_channels+3, *out_channels]
        for i in range(1, len(out_channels)):
            layers += [nn.Conv2d(out_channels[i - 1], out_channels[i], 1, bias=True), nn.BatchNorm2d(out_channels[i], eps=0.001), nn.ReLU()]
        self.conv = nn.Sequential(*layers)
        
    def forward(self, points, features):
        new_points = self.sample(points)
        new_features = self.group(points, new_points, features)
        new_features = self.conv(new_features)
        new_features = new_features.max(dim=3)[0]
        return new_points, new_features
    
class FlowEmbedding(nn.Module):
    def __init__(self, num_samples, in_channels, out_channels):
        super(FlowEmbedding, self).__init__()
        
        self.num_samples = num_samples
        
        self.group = Group(None, self.num_samples, knn=True)
        
        layers = []
        out_channels = [2*in_channels+3, *out_channels]
        for i in range(1, len(out_channels)):
            layers += [nn.Conv2d(out_channels[i - 1], out_channels[i], 1, bias=True), nn.BatchNorm2d(out_channels[i], eps=0.001), nn.ReLU()]
        self.conv = nn.Sequential(*layers)
        
    def forward(self, points1, points2, features1, features2):
        new_features = self.group(points2, points1, features2)
        new_features = torch.cat([new_features, features1.unsqueeze(3).expand(-1, -1, -1, self.num_samples)], dim=1)
        new_features = self.conv(new_features)
        new_features = new_features.max(dim=3)[0]
        return new_features
    
class SetUpConv(nn.Module):
    def __init__(self, num_samples, in_channels1, in_channels2, out_channels1, out_channels2):
        super(SetUpConv, self).__init__()
        
        self.group = Group(None, num_samples, knn=True)
        
        layers = []
        out_channels1 = [in_channels1+3, *out_channels1]
        for i in range(1, len(out_channels1)):
            layers += [nn.Conv2d(out_channels1[i - 1], out_channels1[i], 1, bias=True), nn.BatchNorm2d(out_channels1[i], eps=0.001), nn.ReLU()]
        self.conv1 = nn.Sequential(*layers)
        
        layers = []
        if len(out_channels1) == 1:
            out_channels2 = [in_channels1+in_channels2+3, *out_channels2]
        else:
            out_channels2 = [out_channels1[-1]+in_channels2, *out_channels2]
        for i in range(1, len(out_channels2)):
            layers += [nn.Conv2d(out_channels2[i - 1], out_channels2[i], 1, bias=True), nn.BatchNorm2d(out_channels2[i], eps=0.001), nn.ReLU()]
        self.conv2 = nn.Sequential(*layers)
        
    def forward(self, points1, points2, features1, features2):
        new_features = self.group(points1, points2, features1)
        new_features = self.conv1(new_features)
        new_features = new_features.max(dim=3)[0]
        new_features = torch.cat([new_features, features2], dim=1)
        new_features = new_features.unsqueeze(3)
        new_features = self.conv2(new_features)
        new_features = new_features.squeeze(3)
        return new_features
    
class FeaturePropagation(nn.Module):
    def __init__(self, in_channels1, in_channels2, out_channels):
        super(FeaturePropagation, self).__init__()
        
        layers = []
        out_channels = [in_channels1+in_channels2, *out_channels]
        for i in range(1, len(out_channels)):
            layers += [nn.Conv2d(out_channels[i - 1], out_channels[i], 1, bias=True), nn.BatchNorm2d(out_channels[i], eps=0.001), nn.ReLU()]
        self.conv = nn.Sequential(*layers)
        
    def forward(self, points1, points2, features1, features2):
        dist, ind = three_nn(points2.permute(0, 2, 1).contiguous(), points1.permute(0, 2, 1).contiguous())
        dist = dist * dist
        dist[dist < 1e-10] = 1e-10
        inverse_dist = 1.0 / dist
        norm = torch.sum(inverse_dist, dim=2, keepdim=True)
        weights = inverse_dist / norm
        #new_features = three_interpolate(features1, ind, weights) # wrong gradients
        new_features = torch.sum(group_gather_by_index(features1, ind) * weights.unsqueeze(1), dim = 3)
        new_features = torch.cat([new_features, features2], dim=1)
        new_features = self.conv(new_features.unsqueeze(3)).squeeze(3)
        return new_features

class FlowNet3D(nn.Module):
    def __init__(self):
        super(FlowNet3D, self).__init__()
        
        self.set_conv1 = SetConv(1024, 0.5, 16, 3, [32, 32, 64])
        self.set_conv2 = SetConv(256, 1.0, 16, 64, [64, 64, 128])
        self.flow_embedding = FlowEmbedding(64, 128, [128, 128, 128])
        self.set_conv3 = SetConv(64, 2.0, 8, 128, [128, 128, 256])
        self.set_conv4 = SetConv(16, 4.0, 8, 256, [256, 256, 512])
        self.set_upconv1 = SetUpConv(8, 512, 256, [], [256, 256])
        self.set_upconv2 = SetUpConv(8, 256, 256, [128, 128, 256], [256])
        self.set_upconv3 = SetUpConv(8, 256, 64, [128, 128, 256], [256])
        self.fp = FeaturePropagation(256, 3, [256, 256])
        self.classifier = nn.Sequential(
            nn.Conv1d(256, 128, 1, bias=True),
            nn.BatchNorm1d(128, eps=0.001),
            nn.ReLU(),
            nn.Conv1d(128, 3, 1, bias=True)
        )
         
    def forward(self, points1, points2, features1, features2):
        points1_1, features1_1 = self.set_conv1(points1, features1)
        points1_2, features1_2 = self.set_conv2(points1_1, features1_1)

        points2_1, features2_1 = self.set_conv1(points2, features2)
        points2_2, features2_2 = self.set_conv2(points2_1, features2_1)

        embedding = self.flow_embedding(points1_2, points2_2, features1_2, features2_2)
        
        points1_3, features1_3 = self.set_conv3(points1_2, embedding)
        points1_4, features1_4 = self.set_conv4(points1_3, features1_3)
        
        new_features1_3 = self.set_upconv1(points1_4, points1_3, features1_4, features1_3)
        new_features1_2 = self.set_upconv2(points1_3, points1_2, new_features1_3, torch.cat([features1_2, embedding], dim=1))
        new_features1_1 = self.set_upconv3(points1_2, points1_1, new_features1_2, features1_1)
        new_features1 = self.fp(points1_1, points1, new_features1_1, features1)

        flow = self.classifier(new_features1)
        
        return flow


class Kitti_FlowNet3D_backups(nn.Module):
    def __init__(self):
        super(Kitti_FlowNet3D, self).__init__()

        self.set_conv1 = SetConv(8192, 0.5, 256, 3, [32, 32, 64])
        self.set_conv2 = SetConv(2048, 1.0, 256, 64, [64, 64, 128])
        self.flow_embedding = FlowEmbedding(256, 128, [128, 128, 128])
        self.set_conv3 = SetConv(512, 2.0, 64, 128, [128, 128, 256])
        self.set_conv4 = SetConv(256, 4.0, 64, 256, [256, 256, 512])
        self.set_upconv1 = SetUpConv(4, 512, 256, [], [256, 256])
        self.set_upconv2 = SetUpConv(4, 256, 256, [128, 128, 256], [256])
        self.set_upconv3 = SetUpConv(4, 256, 64, [128, 128, 256], [256])
        self.fp = FeaturePropagation(256, 3, [256, 256])
        self.classifier = nn.Sequential(
            nn.Conv1d(256, 128, 1, bias=True),
            nn.BatchNorm1d(128, eps=0.001),
            nn.ReLU(),
            nn.Conv1d(128, 3, 1, bias=True)
        )

    def forward(self, points1, points2, features1, features2):
        points1_1, features1_1 = self.set_conv1(points1, features1)
        points1_2, features1_2 = self.set_conv2(points1_1, features1_1)

        points2_1, features2_1 = self.set_conv1(points2, features2)
        points2_2, features2_2 = self.set_conv2(points2_1, features2_1)

        embedding = self.flow_embedding(points1_2, points2_2, features1_2, features2_2)

        points1_3, features1_3 = self.set_conv3(points1_2, embedding)
        points1_4, features1_4 = self.set_conv4(points1_3, features1_3)

        new_features1_3 = self.set_upconv1(points1_4, points1_3, features1_4, features1_3)
        new_features1_2 = self.set_upconv2(points1_3, points1_2, new_features1_3,
                                           torch.cat([features1_2, embedding], dim=1))
        new_features1_1 = self.set_upconv3(points1_2, points1_1, new_features1_2, features1_1)
        new_features1 = self.fp(points1_1, points1, new_features1_1, features1)

        flow = self.classifier(new_features1)

        return flow

class Kitti_FlowNet3D(nn.Module):
    def __init__(self):
        super(Kitti_FlowNet3D, self).__init__()

        self.set_conv1 = SetConv(2048, 0.5, 128, 3, [32, 32, 64])
        self.set_conv2 = SetConv(512, 1.0, 128, 64, [64, 64, 128])
        self.flow_embedding = FlowEmbedding(128, 128, [128, 128, 128])
        self.set_conv3 = SetConv(128, 2.0, 32, 128, [128, 128, 256])
        self.set_conv4 = SetConv(32, 4.0, 32, 256, [256, 256, 512])
        self.set_upconv1 = SetUpConv(4, 512, 256, [], [256, 256])
        self.set_upconv2 = SetUpConv(4, 256, 256, [128, 128, 256], [256])
        self.set_upconv3 = SetUpConv(4, 256, 64, [128, 128, 256], [256])
        self.fp = FeaturePropagation(256, 3, [256, 256])
        self.classifier = nn.Sequential(
            nn.Conv1d(256, 128, 1, bias=True),
            nn.BatchNorm1d(128, eps=0.001),
            nn.ReLU(),
            nn.Conv1d(128, 3, 1, bias=True)
        )

    def forward(self, points1, points2, features1, features2):
        points1_1, features1_1 = self.set_conv1(points1, features1)
        points1_2, features1_2 = self.set_conv2(points1_1, features1_1)

        points2_1, features2_1 = self.set_conv1(points2, features2)
        points2_2, features2_2 = self.set_conv2(points2_1, features2_1)

        embedding = self.flow_embedding(points1_2, points2_2, features1_2, features2_2)

        points1_3, features1_3 = self.set_conv3(points1_2, embedding)
        points1_4, features1_4 = self.set_conv4(points1_3, features1_3)

        new_features1_3 = self.set_upconv1(points1_4, points1_3, features1_4, features1_3)
        new_features1_2 = self.set_upconv2(points1_3, points1_2, new_features1_3,
                                           torch.cat([features1_2, embedding], dim=1))
        new_features1_1 = self.set_upconv3(points1_2, points1_1, new_features1_2, features1_1)
        new_features1 = self.fp(points1_1, points1, new_features1_1, features1)

        flow = self.classifier(new_features1)

        return flow

# data
net = Kitti_FlowNet3D().cuda()
net.load_state_dict(torch.load("models/net.pth"))
if torch.cuda.device_count() > 1:
    net = nn.DataParallel(net)
    print("Let's use", torch.cuda.device_count(), "GPUs!")
#

# parameters
BATCH_SIZE = 4
NUM_POINTS = 4096
NUM_EPOCHS = 300
INIT_LR = 0.00001
MIN_LR = 0.00000001
STEP_SIZE_LR = 10
GAMMA_LR = 0.3
INIT_BN_MOMENTUM = 0.2
MIN_BN_MOMENTUM = 0.1
STEP_SIZE_BN_MOMENTUM = 10
GAMMA_BN_MOMENTUM = 0.5

# data
train_manager = Manager()
train_cache = train_manager.dict()
train_dataset = Kitti_SceneflowDataset(npoints=NUM_POINTS, train=True, cache=train_cache)
train_loader = DataLoader(train_dataset,
                          batch_size=BATCH_SIZE,
                          num_workers=4,
                          shuffle=True,
                          pin_memory=True,
                          drop_last=True)
print('train:', len(train_dataset), '/', len(train_loader))

test_manager = Manager()
test_cache = test_manager.dict()
test_dataset = Kitti_SceneflowDataset(npoints=NUM_POINTS, train=False, cache=test_cache)
test_loader = DataLoader(test_dataset,
                        batch_size=int(BATCH_SIZE/2),
                        num_workers=4,
                        pin_memory=True,
                        drop_last=True)
print('test:', len(test_dataset), '/', len(test_loader))

# net
def init_weights(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1.0)
        m.bias.data.fill_(0.0)

#net = FlowNet3D().cuda()
#net.apply(init_weights)
print('# parameters: ', parameter_count(net))

# optimizer
optimizer = optim.Adam(net.parameters(), lr=INIT_LR)

# learning rate scheduler
lr_scheduler = ClippedStepLR(optimizer, STEP_SIZE_LR, MIN_LR, GAMMA_LR)

# batch norm momentum scheduler
def update_bn_momentum(epoch):
    for m in net.modules():
        if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
            m.momentum = max(INIT_BN_MOMENTUM * GAMMA_BN_MOMENTUM ** (epoch // STEP_SIZE_BN_MOMENTUM), MIN_BN_MOMENTUM)

# statistics
losses_train = []
losses_test = []

best_epe=np.inf
# for num_epochs
for epoch in range(NUM_EPOCHS):
    
    # update batch norm momentum
    update_bn_momentum(epoch)
    
    # train mode
    net.train()
    # fix batchnorm
    def fix_bn(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.eval()
    net.apply(fix_bn)  # fix batchnorm
    # statistics
    running_loss = 0.0
    torch.cuda.synchronize()
    start_time = time.time()
    # for each mini-batch
    for points1, points2, features1, features2, flow, mask1 in train_loader:
        # to GPU
        points1 = points1.cuda(non_blocking=True)
        points2 = points2.cuda(non_blocking=True)
        features1 = features1.cuda(non_blocking=True)
        features2 = features2.cuda(non_blocking=True)
        flow = flow.cuda(non_blocking=True)
        mask1 = mask1.cuda(non_blocking=True)
        
        # zero the parameter gradients
        optimizer.zero_grad()
        
        # forward + backward + optimize
        pred_flow = net(points1, points2, features1, features2)
        loss = criterion(pred_flow, flow, mask1)
        loss.backward()
        optimizer.step()
        
        # statistics
        running_loss += loss.item()
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    running_loss /= (len(train_loader))
    
    losses_train.append(running_loss)
    
    # output
    print('Epoch {} (train) -- loss: {:.6f} -- duration (epoch/iteration): {:.4f} min/{:.4f} sec'.format(epoch, running_loss, (end_time-start_time)/60.0, (end_time-start_time)/len(train_loader)))
    
    loss_sum = 0
    epe_sum = 0
    acc050_sum = 0
    acc010_sum = 0
    #
    with torch.no_grad():
        # for each mini-batch
        net.eval()
        for points1, points2, features1, features2, flow, mask1 in test_loader:
            #
            #         # to GPU
            points1 = points1.cuda(non_blocking=True)
            points2 = points2.cuda(non_blocking=True)
            features1 = features1.cuda(non_blocking=True)
            features2 = features2.cuda(non_blocking=True)
            flow = flow.cuda(non_blocking=True)
            mask1 = mask1.cuda(non_blocking=True)
            #
            pred_flow_sum = torch.zeros(2, 3, 4096).cuda(non_blocking=True)
            #
            #         # resample 10 times
            for i in range(10):
                #
                perm = torch.randperm(points1.shape[2])
                points1_perm = points1[:, :, perm]
                points2_perm = points2[:, :, perm]
                features1_perm = features1[:, :, perm]
                features2_perm = features2[:, :, perm]
                #
                #             # forward
                pred_flow = net(points1_perm, points2_perm, features1_perm, features2_perm)
                pred_flow_sum[:, :, perm] += pred_flow
            #
            #         # statistics
            pred_flow_sum /= 10
            loss = criterion(pred_flow_sum, flow, mask1)
            loss_sum += loss.item()
            epe, acc050, acc010 = error(pred_flow_sum, flow, mask1)
            epe_sum += epe
            acc050_sum += acc050
            acc010_sum += acc010# validate
            
        epe_sum /= len(test_loader)
        losses_test.append(epe_sum)

        # output
        print('Epoch {} (test) -- epe: {:.6f} -- duration (epoch/iteration): {:.4f} min/{:.4f} sec'.format(epoch, epe_sum, (end_time-start_time)/60.0, (end_time-start_time)/len(train_loader)))
        
        if epe_sum < best_epe:
          best_epe = epe_sum
          if torch.cuda.device_count() > 1:
              torch.save(net.module.state_dict(), 'kitti_models_rm_ground_v2/net_epe_%.4f.pth' % epe_sum)
          else:
              torch.save(net.state_dict(),'kitti_models_rm_ground_v2/net_epe_%.4f.pth' % epe_sum)
          '''
          net_reload = Kitti_FlowNet3D().cuda()
          net_reload.load_state_dict(torch.load('kitti_models_rm_ground_v2/net_epe_%.4f.pth' % epe_sum))
          loss_sum = 0
          epe_sum = 0
          acc050_sum = 0
          acc010_sum = 0
          with torch.no_grad():
              # for each mini-batch
              net_reload = nn.DataParallel(net_reload)
              net_reload.eval()
              for points1, points2, features1, features2, flow, mask1 in test_loader:
                  #
                  #         # to GPU
                  points1 = points1.cuda(non_blocking=True)
                  points2 = points2.cuda(non_blocking=True)
                  features1 = features1.cuda(non_blocking=True)
                  features2 = features2.cuda(non_blocking=True)
                  flow = flow.cuda(non_blocking=True)
                  mask1 = mask1.cuda(non_blocking=True)
                  #
                  pred_flow_sum = torch.zeros(2, 3, 4096).cuda(non_blocking=True)
                  #
                  #         # resample 10 times
                  for i in range(10):
                      #
                      perm = torch.randperm(points1.shape[2])
                      points1_perm = points1[:, :, perm]
                      points2_perm = points2[:, :, perm]
                      features1_perm = features1[:, :, perm]
                      features2_perm = features2[:, :, perm]
                      #
                      #             # forward
                      pred_flow = net_reload(points1_perm, points2_perm, features1_perm, features2_perm)
                      pred_flow_sum[:, :, perm] += pred_flow
                  #
                  #         # statistics
                  pred_flow_sum /= 10
                  loss = criterion(pred_flow_sum, flow, mask1)
                  loss_sum += loss.item()
                  epe, acc050, acc010 = error(pred_flow_sum, flow, mask1)
                  epe_sum += epe
                  acc050_sum += acc050
                  acc010_sum += acc010  # validate

              epe_sum /= len(test_loader)
              losses_test.append(epe_sum)

              # output
              print('RELOAD: Epoch {} (test) -- epe: {:.6f} -- duration (epoch/iteration): {:.4f} min/{:.4f} sec'.format(epoch, epe_sum, (end_time-start_time)/60.0, (end_time-start_time)/len(train_loader)))
    '''
  # update learning rate
    lr_scheduler.step()
    
    print('---')
    
#plt.plot(losses_train)
#plt.plot(losses_test)
f_train=open('kitti_losses_train.txt','wb')
f_test=open('kitti_losses_test.txt','wb')
pickle.dump(losses_train,f_train)
pickle.dump(losses_test,f_test)
f_train.close()
f_test.close()
#net = net.cpu()
torch.save(net.module.state_dict(),'kitti_models_rm_ground_v2/net.pth')

# Commented out IPython magic to ensure Python compatibility.
# %%time
# 
# # data
test_set = Kitti_SceneflowDataset(train=False)
test_loader = DataLoader(test_set,
                         batch_size=2,
                         num_workers=4,
                         pin_memory=True,
                         drop_last=True)

print('test set:', len(test_set), 'samples /', len(test_loader), 'mini-batches')
# 
# # model
net = Kitti_FlowNet3D().cuda()
net.load_state_dict(torch.load('kitti_models_rm_ground_v2/net.pth'))
net.eval()
# 
# # statistics
loss_sum = 0
epe_sum = 0
acc050_sum = 0
acc010_sum = 0
# 
with torch.no_grad():

    # for each mini-batch
    for points1, points2, features1, features2, flow, mask1 in test_loader:
#             
#         # to GPU
        points1 = points1.cuda(non_blocking=True)
        points2 = points2.cuda(non_blocking=True)
        features1 = features1.cuda(non_blocking=True)
        features2 = features2.cuda(non_blocking=True)
        flow = flow.cuda(non_blocking=True)
        mask1 = mask1.cuda(non_blocking=True)
#     
        pred_flow_sum = torch.zeros(2, 3, 4096).cuda(non_blocking=True)
#         
#         # resample 10 times
        for i in range(10):
#             
            perm = torch.randperm(points1.shape[2])
            points1_perm = points1[:, :, perm]
            points2_perm = points2[:, :, perm]
            features1_perm = features1[:, :, perm]
            features2_perm = features2[:, :, perm]
# 
#             # forward
            pred_flow = net(points1_perm, points2_perm, features1_perm, features2_perm)
            pred_flow_sum[:, :, perm] += pred_flow
#         
#         # statistics
        pred_flow_sum /= 10
        loss = criterion(pred_flow_sum, flow, mask1)
        loss_sum += loss.item()
        epe, acc050, acc010 = error(pred_flow_sum, flow, mask1)
        epe_sum += epe
        acc050_sum += acc050
        acc010_sum += acc010
#         
print('mean loss:', loss_sum/len(test_loader))
print('mean epe:', epe_sum/len(test_loader))
print('mean acc050:', acc050_sum/len(test_loader))
print('mean acc010:', acc010_sum/len(test_loader))
#     
print('---')

f=open('result.txt','wb')
pickle.dump({'mean loss':loss_sum/len(test_loader),'mean epe':epe_sum/len(test_loader),\
             'mean acc050': acc050_sum/len(test_loader),'mean acc010': acc010_sum/len(test_loader)},f)