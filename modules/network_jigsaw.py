import torch.nn as nn
import torch
from torch.nn.functional import normalize


class Network(nn.Module):
    def __init__(self, resnet, feature_dim, class_num):
        super(Network, self).__init__()
        self.resnet = resnet
        self.feature_dim = feature_dim
        self.cluster_num = class_num
        self.instance_projector = nn.Sequential(
            nn.Linear(self.resnet.rep_dim, self.resnet.rep_dim),
            nn.ReLU(),
            nn.Linear(self.resnet.rep_dim, self.feature_dim),
        )
        self.cluster_projector = nn.Sequential(
            nn.Linear(self.resnet.rep_dim, self.resnet.rep_dim,),
            nn.ReLU(),
            nn.Linear(self.resnet.rep_dim, self.cluster_num),
            nn.Softmax(dim=1)
        )
        self.jigsaw_projector = nn.Sequential(
            nn.Linear(self.resnet.rep_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 6),
            nn.Softmax(dim=1)
        )

    def forward(self, x_i, x_j, x_jigsaw):
        h_a = self.resnet(x_i)
        h_b = self.resnet(x_j)
        h_bj = self.resnet(x_jigsaw)

        z_a = normalize(self.instance_projector(h_a), dim=1)
        z_b = normalize(self.instance_projector(h_b), dim=1)

        y_a = self.cluster_projector(h_a)
        y_b = self.cluster_projector(h_b)

        v = self.jigsaw_projector(h_bj)

        return z_a, z_b, y_a, y_b, v

    def forward_cluster(self, x):
        h = self.resnet(x)
        c = self.cluster_projector(h)
        c2 = self.instance_projector(h)
        c = torch.argmax(c, dim=1)
        return c, c2
