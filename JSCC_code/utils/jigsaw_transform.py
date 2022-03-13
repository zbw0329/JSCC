import torch
import random

from utils import labels_switch


def random_dict(dicts):
    dict_key_ls = list(dicts.keys())
    random.shuffle(dict_key_ls)
    new_dic = {}
    for key in dict_key_ls:
        new_dic[key] = dicts.get(key)
    return new_dic


def jigsaw_transform(q):
    c1, c2 = q.split([112, 112], dim=2)
    f0, f1 = c1.split([112, 112], dim=3)
    f2, f3 = c2.split([112, 112], dim=3)

    image_dict = {0: f0, 1: f1, 2: f2, 3: f3}
    image_dict = random_dict(image_dict)
    image_labels = []
    labels = ""
    for k, v in image_dict.items():
        image_labels.append(k)
        labels = "".join([labels, str(k)])

    image_gather1 = torch.cat([image_dict.get(image_labels[0]), image_dict.get(image_labels[1])], dim=3)
    image_gather2 = torch.cat([image_dict.get(image_labels[2]), image_dict.get(image_labels[3])], dim=3)
    image_gather = torch.cat([image_gather1, image_gather2], dim=2)
    labels = labels_switch.labels_switch(labels)
    labels = torch.tensor([labels] * q.shape[0])
    labels = labels.view(-1)
    labels = torch.as_tensor(labels)

    return image_gather, labels
