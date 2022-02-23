# -*- coding:utf-8 -*-

# @Filename: my_modules
# @Project : Unsupervised_Domian_Adaptation
# @date    : 2021-11-27 12:49
# @Author  : Linshan
import torch
import math
import torch.nn as nn
import torch.nn.functional as F


def one_hot(label, num_classes=7):
    label = F.one_hot(label, num_classes=num_classes + 1)
    label = label.permute(0, 3, 1, 2)
    return label[:, :-1, :, :]


def VectorPooling(feat, pred, label, num_classes=7, ignore_index=-1):
    B, C, H, W = feat.size()
    _, num_cls, _, _ = pred.size()

    # delete the ignore index
    label_copy = label.clone()
    label_copy[label == ignore_index] = num_classes

    label = one_hot(label_copy, num_classes=num_classes)
    pred = pred.softmax(dim=1).detach()
    anchor = label * pred
    anchor = F.interpolate(anchor, size=(H, W), mode='bilinear', align_corners=True)\
        .contiguous().view(num_cls, 1, B*H*W)

    feat = feat.view(1, C, B*H*W)
    vectors = (feat * anchor).sum(-1) / anchor.sum(-1)  # (num_cls, C)
    return vectors[1:, :]  # ignore the background


def del_tensor(tensor, index):
    arr1 = tensor[0:index, :]
    arr2 = tensor[index + 1:, :]
    return torch.cat((arr1, arr2), dim=0)


def cal_dist(p, z, version='simplified'):
    if version == 'original':
        p = F.normalize(p, dim=1)  # l2-normalize
        z = F.normalize(z, dim=1)  # l2-normalize
        return 1 - (p * z).sum(dim=1).mean()

    elif version == 'simplified':  # same thing, much faster. Scroll down, speed test in __main__
        p = F.normalize(p, dim=1)  # l2-normalize
        z = F.normalize(z, dim=1)  # l2-normalize
        return 1 - F.cosine_similarity(p, z, dim=-1).mean()
    else:
        raise Exception


def anchor_triplet_loss(vectors1, vectors2, alpha=0.5):
    num_cls, C = vectors1.size()  # num_cls = num_classes - 1
    assert vectors1.size() == vectors2.size(), print(vectors1.size(), vectors2.size())

    loss_sum = 0
    for i in range(num_cls):
        vec = vectors2[i, :].view(1, C)

        # positive distance
        pos = cal_dist(vec, vectors1[i, :].view(1, C)).mean()
        # negative distance
        # new_vectors1 = del_tensor(vectors1, i)
        # neg = cal_dist(vec, new_vectors1)

        new_loss = pos
        loss_sum += new_loss

    loss_sum = loss_sum/num_cls
    return loss_sum


if __name__ == '__main__':
    label = torch.randint(low=-1, high=7, size=(2, 512, 512)).long()
    # label = F.one_hot(label, num_classes=7)
    # label = label.permute(0, 3, 1, 2)

    feat = torch.rand(2, 1024, 32, 32)
    pred = torch.rand(2, 7, 512, 512)

    vecs = VectorPooling(feat, pred, label)
    loss = anchor_triplet_loss(vecs, vecs)
    print(loss)

