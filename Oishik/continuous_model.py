import numpy as np
import torch
import sys

np.set_printoptions(threshold=sys.maxsize)


def potential(theta, alpha, l, s, y, k):
    eq = (2 * torch.eq(k.view(-1, 1).double(), y.double()).double() - 1).t()
    return torch.tensordot(torch.sigmoid(s.double() - alpha).double() * torch.abs(l.double()), eq.mul(theta.double()).squeeze(), 1)


def un_normalized(theta, alpha, l, s, y, k):
    return torch.exp(potential(theta, alpha, l, s, y, k))


def probabilities(theta, alpha, l, s, k, n_classes, n_lfs):
    prob = torch.ones(l.shape[0], n_classes)
    z = calculate_z(theta, alpha, k, n_classes, n_lfs)
    for y in range(n_classes):
        y_ = torch.tensor(y)
        prob[:, y] = un_normalized(theta, alpha, l, s, y_, k) / z

    return prob


def calculate_z(theta, alpha, lf_classes, n_classes=2, n_lfs=2, bin_width=0.1):
    s = torch.arange(0, 1, bin_width)
    s = s.view(-1, 1) * torch.ones(n_lfs)
    z = 0
    for y in range(n_classes):
        y = torch.tensor(y)
        eq = (2 * torch.eq(lf_classes.view(-1, 1).double(), y.double()).double() - 1).t()
        M_y = torch.sigmoid(s.double() - alpha).double() * eq.mul(theta.double()).squeeze()
        z += (1 + torch.exp(M_y).sum(0) * bin_width).prod()
    return z


def get_loss(theta, alpha, l, s, k, n_classes, n_lfs):
    z = calculate_z(theta, alpha, k, n_classes, n_lfs)
    probability = 0
    for y in range(n_classes):
        y = torch.tensor([y])
        probability += un_normalized(theta, alpha, l, s, y, k) / z
    log_probability = torch.log(probability)

    return - torch.sum(log_probability)


def expectation_step(theta, alpha, l, s, k, n_classes, n_lfs):
    q = probabilities(theta, alpha, l, s, k, n_classes, n_lfs)
    q = q / q.sum(1).view(-1, 1)
    return q


def maximization_step(theta, alpha, l, s, k, n_classes, n_lfs):
    q = expectation_step(theta, alpha, l, s, k, n_classes, n_lfs)
    p = probabilities(theta, alpha, l, s, k, n_classes, n_lfs)
    return q*torch.log(p)


def em_loss(theta, alpha, l, s, k, n_classes, n_lfs):
    return - maximization_step(theta, alpha, l, s, k, n_classes, n_lfs).sum()

