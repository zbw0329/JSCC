import torchvision
from modules import transform
from torch.utils import data
import torch


def make_dataset(args):
    # make dataset for training
    in_channels = 3
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
        dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset, unlabeled_dataset])
        class_num = 10
    elif args.dataset == "CIFAR-10":
        train_dataset = torchvision.datasets.CIFAR10(
            root=args.dataset_dir,
            download=True,
            train=True,
            transform=transform.Transforms(size=args.image_size, s=0.5),
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root=args.dataset_dir,
            download=True,
            train=False,
            transform=transform.Transforms(size=args.image_size, s=0.5),
        )
        dataset = data.ConcatDataset([train_dataset, test_dataset])
        class_num = 10
    elif args.dataset == "CIFAR-100":
        train_dataset = torchvision.datasets.CIFAR100(
            root=args.dataset_dir,
            download=True,
            train=True,
            transform=transform.Transforms(size=args.image_size, s=0.5),
        )
        test_dataset = torchvision.datasets.CIFAR100(
            root=args.dataset_dir,
            download=True,
            train=False,
            transform=transform.Transforms(size=args.image_size, s=0.5),
        )
        dataset = data.ConcatDataset([train_dataset, test_dataset])
        class_num = 20
    elif args.dataset == "ImageNet-10":
        dataset = torchvision.datasets.ImageFolder(
            root=args.dataset_dir + '/ImageNet-10/train',
            transform=transform.Transforms(size=args.image_size, blur=True),
            # transform=transform.Transforms(size=args.image_size).test_transform,
        )
        class_num = 10
    elif args.dataset == "ImageNet-dogs":
        dataset = torchvision.datasets.ImageFolder(
            root=args.dataset_dir + '/ImageNet-dogs/train',
            transform=transform.Transforms(size=args.image_size, blur=True),
            # transform=transform.Transforms(size=args.image_size).test_transform,
        )
        class_num = 15
    elif args.dataset == "tiny-ImageNet":
        dataset = torchvision.datasets.ImageFolder(
            root=args.dataset_dir + '/tiny-imagenet-200/train',
            transform=transform.Transforms(s=0.5, size=args.image_size),
        )
        class_num = 200
    elif args.dataset == "USPS":
        train_dataset = torchvision.datasets.USPS(
            root=args.dataset_dir,
            download=True,
            train=True,
            transform=transform.Transforms(size=args.image_size, s=0.5),
        )
        test_dataset = torchvision.datasets.USPS(
            root=args.dataset_dir,
            download=True,
            train=False,
            transform=transform.Transforms(size=args.image_size, s=0.5),
        )
        dataset = data.ConcatDataset([train_dataset, test_dataset])
        class_num = 10
        in_channels = 1
    elif args.dataset == "FMNIST":
        train_dataset = torchvision.datasets.FashionMNIST(
            root=args.dataset_dir,
            download=True,
            train=True,
            transform=transform.Transforms(size=args.image_size, s=0.5),
        )
        test_dataset = torchvision.datasets.FashionMNIST(
            root=args.dataset_dir,
            download=True,
            train=False,
            transform=transform.Transforms(size=args.image_size, s=0.5),
        )
        dataset = data.ConcatDataset([train_dataset, test_dataset])
        class_num = 10
        in_channels = 1
    elif args.dataset == "MNIST":
        train_dataset = torchvision.datasets.MNIST(
            root=args.dataset_dir,
            download=True,
            train=True,
            transform=transform.Transforms(size=args.image_size, s=0.5),
        )
        test_dataset = torchvision.datasets.MNIST(
            root=args.dataset_dir,
            download=True,
            train=False,
            transform=transform.Transforms(size=args.image_size, s=0.5),
        )
        dataset = data.ConcatDataset([train_dataset, test_dataset])
        class_num = 10
        in_channels = 1
    else:
        raise NotImplementedError
    return dataset, class_num, in_channels


def make_dataset_eval(args):
    # make dataset for clustering
    in_channels = 3
    if args.dataset == "CIFAR-10":
        train_dataset = torchvision.datasets.CIFAR10(
            root=args.dataset_dir,
            train=True,
            download=True,
            transform=transform.Transforms(size=args.image_size).test_transform,
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root=args.dataset_dir,
            train=False,
            download=True,
            transform=transform.Transforms(size=args.image_size).test_transform,
        )
        dataset = data.ConcatDataset([train_dataset, test_dataset])
        class_num = 10
    elif args.dataset == "CIFAR-100":
        train_dataset = torchvision.datasets.CIFAR100(
            root=args.dataset_dir,
            download=True,
            train=True,
            transform=transform.Transforms(size=args.image_size).test_transform,
        )
        test_dataset = torchvision.datasets.CIFAR100(
            root=args.dataset_dir,
            download=True,
            train=False,
            transform=transform.Transforms(size=args.image_size).test_transform,
        )
        dataset = data.ConcatDataset([train_dataset, test_dataset])
        class_num = 20
    elif args.dataset == "STL-10":
        train_dataset = torchvision.datasets.STL10(
            root=args.dataset_dir,
            split="train",
            download=True,
            transform=transform.Transforms(size=args.image_size).test_transform,
        )
        test_dataset = torchvision.datasets.STL10(
            root=args.dataset_dir,
            split="test",
            download=True,
            transform=transform.Transforms(size=args.image_size).test_transform,
        )
        dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
        class_num = 10
    elif args.dataset == "ImageNet-10":
        dataset = torchvision.datasets.ImageFolder(
            root=args.dataset_dir + '/ImageNet-10/train',
            transform=transform.Transforms(size=args.image_size).test_transform,
        )
        class_num = 10
    elif args.dataset == "ImageNet-dogs":
        dataset = torchvision.datasets.ImageFolder(
            root=args.dataset_dir + '/ImageNet-dogs/train',
            transform=transform.Transforms(size=args.image_size).test_transform,
        )
        class_num = 15
    elif args.dataset == "tiny-ImageNet":
        dataset = torchvision.datasets.ImageFolder(
            root=args.dataset_dir + '/tiny-imagenet-200/train',
            transform=transform.Transforms(size=args.image_size).test_transform,
        )
        class_num = 200
    elif args.dataset == "USPS":
        train_dataset = torchvision.datasets.USPS(
            root=args.dataset_dir,
            download=True,
            train=True,
            transform=transform.Transforms(size=args.image_size).test_transform,
        )
        test_dataset = torchvision.datasets.USPS(
            root=args.dataset_dir,
            download=True,
            train=False,
            transform=transform.Transforms(size=args.image_size).test_transform,
        )
        dataset = data.ConcatDataset([train_dataset, test_dataset])
        class_num = 10
        in_channels = 1
    elif args.dataset == "FMNIST":
        train_dataset = torchvision.datasets.FashionMNIST(
            root=args.dataset_dir,
            download=True,
            train=True,
            transform=transform.Transforms(size=args.image_size).test_transform,
        )
        test_dataset = torchvision.datasets.FashionMNIST(
            root=args.dataset_dir,
            download=True,
            train=False,
            transform=transform.Transforms(size=args.image_size).test_transform,
        )
        dataset = data.ConcatDataset([train_dataset, test_dataset])
        class_num = 10
        in_channels = 1
    elif args.dataset == "MNIST":
        train_dataset = torchvision.datasets.MNIST(
            root=args.dataset_dir,
            download=True,
            train=True,
            transform=transform.Transforms(size=args.image_size, s=0.5),
        )
        test_dataset = torchvision.datasets.MNIST(
            root=args.dataset_dir,
            download=True,
            train=False,
            transform=transform.Transforms(size=args.image_size, s=0.5),
        )
        dataset = data.ConcatDataset([train_dataset, test_dataset])
        class_num = 10
        in_channels = 1
    else:
        raise NotImplementedError
    return dataset, class_num, in_channels