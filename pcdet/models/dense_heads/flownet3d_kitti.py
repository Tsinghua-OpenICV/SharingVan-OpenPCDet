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
        super(Kitti_FlowNet3D_backups, self).__init__()

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