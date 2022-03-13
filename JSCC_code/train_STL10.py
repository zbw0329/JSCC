import os
import numpy as np
import torch
import torchvision
import argparse
from modules import transform, resnet, network, loss
from utils import yaml_config_hook, save_model
from torch.utils import data
from utils.jigsaw_transform import jigsaw_transform
import time
import datetime


def train():
    loss_epoch = 0
    for step, ((x_i, x_j), _) in enumerate(instance_data_loader):
        optimizer.zero_grad()
        x_i = x_i.to('cuda:0')
        x_j = x_j.to('cuda:0')

        z_i, z_j, c_i, c_j, z = model(x_i, x_j, x_j)
        loss_instance = criterion_instance(z_i, z_j)
        loss = loss_instance
        loss.backward()
        optimizer.step()
        if step % 50 == 0:
            print(
                f"Step [{step}/{len(instance_data_loader)}]\t loss_instance: {loss_instance.item()}")
        loss_epoch += loss.item()

    for step, ((x_i, x_j), _) in enumerate(cluster_data_loader):
        optimizer.zero_grad()
        x_i = x_i.to('cuda:0')
        x_j = x_j.to('cuda:0')
        x_jigsaw, image_labels = jigsaw_transform(x_i)

        image_labels = torch.as_tensor(image_labels)
        image_labels = image_labels.to('cuda')
        z_i, z_j, c_i, c_j, z = model(x_i, x_j, x_jigsaw)
        loss_jigsaw = criterion_jigsaw(z, image_labels.long())
        loss_instance = criterion_instance(z_i, z_j)
        loss_cluster = criterion_cluster(c_i, c_j)
        loss = loss_instance + loss_cluster + loss_jigsaw
        loss.backward()
        optimizer.step()
        if step % 50 == 0:
            print(
                f"Step [{step}/{len(cluster_data_loader)}]\t loss_instance: {loss_instance.item()}\t loss_cluster: {loss_cluster.item()}")
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
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    # prepare data
    if args.dataset == "STL-10":
        train_dataset = torchvision.datasets.STL10(
            root=args.dataset_dir,
            split="train",
            download=True,
            transform=transform.Transforms(size=args.image_size),
        )
        test_dataset = torchvision.datasets.STL10(
            root=args.dataset_dir,
            split="test",
            download=True,
            transform=transform.Transforms(size=args.image_size),
        )
        unlabeled_dataset = torchvision.datasets.STL10(
            root=args.dataset_dir,
            split="unlabeled",
            download=True,
            transform=transform.Transforms(size=args.image_size),
        )
        cluster_dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
        instance_dataset = unlabeled_dataset
        class_num = 10
    else:
        raise NotImplementedError
    cluster_data_loader = torch.utils.data.DataLoader(
        cluster_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
    )
    instance_data_loader = torch.utils.data.DataLoader(
        instance_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
    )

    # initialize model
    res = resnet.get_resnet(args.resnet, 3)
    model = network.Network(res, args.feature_dim, class_num)
    model = model.to('cuda')
    # optimizer / loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    if args.reload:
        model_fp = os.path.join(args.model_path, "{}_checkpoint_{}.tar".format(args.resnet, args.start_epoch))
        checkpoint = torch.load(model_fp)
        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        args.start_epoch = checkpoint['epoch'] + 1
        print(f"{model_fp} has been loaded.")
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
        if epoch % 100 == 0 and epoch != 0:
            save_model(args, model, optimizer, epoch, args.resnet)
        Time = time.time() - Start_Time
        Time_remain = str(datetime.timedelta(seconds=Time * (args.epochs - epoch)))
        print(f"Epoch [{epoch}/{args.epochs}]\t Loss: {loss_epoch / len(instance_data_loader)} Epoch_Time:{Time} Time-Remainig:{Time_remain}")
    save_model(args, model, optimizer, args.epochs, args.resnet)
