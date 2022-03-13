import os
import argparse

import sklearn.cluster
import torch
import torchvision
import numpy as np
from utils import yaml_config_hook, visualization, make_dataset, confusion_matrix
from modules import resnet, network_cluster, transform
from evaluation import evaluation
from torch.utils import data
import torch.nn as nn
import copy
from scipy import io


def inference(loader, model, device):
    model.eval()
    feature_vector = []
    labels_vector = []
    feature_vector_c2 = []
    for step, (x, y) in enumerate(loader):
        x = x.to(device)
        with torch.no_grad():
            c, c2 = model.forward_cluster(x)
        c = c.detach()
        feature_vector.extend(c.cpu().detach().numpy())
        feature_vector_c2.extend(c2.cpu().detach().numpy())
        labels_vector.extend(y.numpy())
        if step % 20 == 0:
            print(f"Step [{step}/{len(loader)}]\t Computing features...")
    feature_vector = np.array(feature_vector)
    labels_vector = np.array(labels_vector)
    # visualization.tSNE(feature_vector_c2, labels_vector, "name")
    print("Features shape {}".format(feature_vector.shape))
    return feature_vector, labels_vector


def make_ResNet(args, class_num):
    res = resnet.get_resnet(args.resnet)
    model = network_cluster.Network(res, args.feature_dim, class_num)
    model_name = "{}_checkpoint_{}.tar".format(args.resnet, args.start_epoch)
    model_fp = os.path.join(args.model_path, model_name)
    checkpoint_dick = torch.load(model_fp, map_location=device.type)['net']
    print(f"the model_fp is {model_fp}")
    return model, checkpoint_dick


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    config = yaml_config_hook("./config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()
    args.seed = np.random.randint(1, 100000)

    gpu_num = args.gpu_num
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num)
    print(f"use gpu:{gpu_num} for clustering")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset, class_num, in_channels = make_dataset.make_dataset_eval(args)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=500,
        shuffle=False,
        drop_last=False,
        num_workers=args.workers,
    )
    model, checkpoint_dick = make_ResNet(args, class_num)
    checkpoint_dick_modified = {k.replace('module.', ''): v for k, v in checkpoint_dick.items()}
    model.load_state_dict(checkpoint_dick_modified, strict=False)
    model.to(device)

    print("### Creating features from model ###")
    X, Y = inference(data_loader, model, device)

    if args.dataset == "CIFAR-100":  # super-class
        super_label = [
            [72, 4, 95, 30, 55],
            [73, 32, 67, 91, 1],
            [92, 70, 82, 54, 62],
            [16, 61, 9, 10, 28],
            [51, 0, 53, 57, 83],
            [40, 39, 22, 87, 86],
            [20, 25, 94, 84, 5],
            [14, 24, 6, 7, 18],
            [43, 97, 42, 3, 88],
            [37, 17, 76, 12, 68],
            [49, 33, 71, 23, 60],
            [15, 21, 19, 31, 38],
            [75, 63, 66, 64, 34],
            [77, 26, 45, 99, 79],
            [11, 2, 35, 46, 98],
            [29, 93, 27, 78, 44],
            [65, 50, 74, 36, 80],
            [56, 52, 47, 59, 96],
            [8, 58, 90, 13, 48],
            [81, 69, 41, 89, 85],
        ]
        Y_copy = copy.copy(Y)
        for i in range(20):
            for j in super_label[i]:
                Y[Y_copy == j] = i
    nmi, ari, f, acc = evaluation.evaluate(Y, X)

    if args.confusion_matrix:
        confusion_matrix.confusion_matrix(X, Y, class_num, args.dataset+'.pdf')
    print('NMI = {:.3f} ACC = {:.3f} ARI = {:.3f} F = {:.3f} '.format(nmi, acc, ari, f))