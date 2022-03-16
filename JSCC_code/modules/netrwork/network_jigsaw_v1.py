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
            nn.Linear(self.resnet.rip_dim, self.resnet.rip_dim),
            nn.ReLU(),
            nn.Linear(self.resnet.rip_dim, self.feature_dim),
        )
        self.cluster_projector = nn.Sequential(
            nn.Linear(self.resnet.rip_dim, self.resnet.rip_dim),
            nn.ReLU(),
            nn.Linear(self.resnet.rip_dim, self.cluster_num),
            nn.Softmax(dim=1)
        )
        self.jigsaw_projector = nn.Sequential(
            nn.Linear(self.resnet.rip_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 6),
            nn.Softmax(dim=1)
        )

    def forward(self, x_i, x_j, x_jigsaw):
        h_i = self.resnet(x_i)
        h_j = self.resnet(x_j)
        h_jigsaw = self.resnet(x_jigsaw)

        z_i = normalize(self.instance_projector(h_i), dim=1)
        z_j = normalize(self.instance_projector(h_j), dim=1)

        c_i = self.cluster_projector(h_i)
        c_j = self.cluster_projector(h_j)

        z = self.jigsaw_projector(h_jigsaw)

        return z_i, z_j, c_i, c_j, z

    def forward_cluster(self, x):
        h = self.resnet(x)
        c = self.cluster_projector(h)
        c2 = self.instance_projector(h)
        c = torch.argmax(c, dim=1)
        return c, c2
