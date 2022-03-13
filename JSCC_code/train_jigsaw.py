import os

import numpy as np
import torch
import argparse

import cluster
from modules import resnet, loss
from modules.netrwork import network_jigsaw_v1 as network
from utils import yaml_config_hook, save_model, make_dataset
from torch.utils import data
import time
from utils.forzen_backbone import forzen_backbone
import datetime
from utils.jigsaw_transform_6patch import jigsaw_transform


def train():
    loss_epoch = 0
    for step, ((x_i, x_j), _) in enumerate(data_loader):
        optimizer.zero_grad()

        x_i = x_i.to('cuda')
        x_j = x_j.to('cuda')
        x_jigsaw, image_labels = jigsaw_transform(x_i)
        x_jigsaw = x_jigsaw.to('cuda')
        image_labels = image_labels.to('cuda')

        z_i, z_j, c_i, c_j, z = model(x_i, x_j, x_jigsaw)

        loss_instance = criterion_instance(z_i, z_j)
        loss_cluster = criterion_cluster(c_i, c_j)
        loss_jigsaw = criterion_jigsaw(z, image_labels.long())

        loss = loss_instance + loss_cluster + loss_jigsaw

        loss.backward()
        optimizer.step()
        if step % 50 == 0:
            print(
                f"Step [{step}/{len(data_loader)}]\t  loss_instance: {loss_instance.item()}\t  "
                f"loss_cluster: {loss_cluster.item()}\t loss_jigsaw: {loss_jigsaw.item()}")
        loss_epoch += loss.item()
    return loss_epoch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    config = yaml_config_hook("config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    gpu_num = args.gpu_num
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num)
    print(f"use gpu:{gpu_num} for training")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"dataset:{args.dataset}\n"
          f"from {args.start_epoch} Epochs to {args.epochs} Epochs\n"
          f"model:{args.resnet}\n"
          f"batch_size:{args.batch_size}\n"
          f"lr:{args.learning_rate}\n")

    # prepare data
    dataset, class_num, in_channels = make_dataset.make_dataset(args)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
        pin_memory=False,
    )


    # initialize model
    res = resnet.get_resnet(args.resnet)
    model = network.Network(res, args.feature_dim, class_num)
    model = torch.nn.DataParallel(model)
    model = model.to('cuda')
    if args.forzen_backbone:
        model = forzen_backbone(model)
    # print(str(model))

    # optimizer / loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    if args.reload:
        model_fp = os.path.join(args.model_path, "{}_checkpoint_{}.tar".format(args.resnet, args.start_epoch))
        checkpoint = torch.load(model_fp)
        model.load_state_dict(checkpoint['net'],strict=False)
        optimizer.load_state_dict(checkpoint['optimizer'])
        args.start_epoch = checkpoint['epoch'] + 1
        print(f"{model_fp} has been loaded")
    else:
        if args.start_epoch != 0:
            print("It must begin with 0 epoch or load a checkpoint for training.")
            raise NotImplementedError
    loss_device = torch.device("cuda")
    criterion_instance = loss.InstanceLoss(args.batch_size, args.instance_temperature, loss_device).to(
        loss_device)
    criterion_cluster = loss.ClusterLoss(class_num, args.cluster_temperature, loss_device).to(loss_device)
    criterion_jigsaw = torch.nn.CrossEntropyLoss().to(loss_device)

    # train
    for epoch in range(args.start_epoch, args.epochs):
        Start_Time = time.time()
        lr = optimizer.param_groups[0]["lr"]
        loss_epoch = train()
        if epoch % 50 == 0:
            save_model(args, model, optimizer, epoch, args.resnet)
        Time = time.time() - Start_Time
        Time_remain = str(datetime.timedelta(seconds=Time * (args.epochs - epoch - 1)))
        print(
            f"Epoch [{epoch}/{args.epochs}]\t Loss: {loss_epoch / len(data_loader)} Epoch_Time:{Time} Time-Remainig:{Time_remain}")
    save_model(args, model, optimizer, args.epochs, args.resnet)