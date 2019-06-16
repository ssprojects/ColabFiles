import numpy as np
import torch


def potential(theta, l, y, k):
    return torch.tensordot((2 * torch.eq(k.view(-1, 1).double(), y.double()).double() - 1).t() * torch.abs(l.double()),
                           theta, 1).double()


def un_normalized(theta, l, y, k):
    return torch.exp(potential(theta, l, y, k))


def calculate_z(theta, lf_classes, n_classes=2):
    z = 0
    for y in range(n_classes):
        y = torch.tensor(y)
        M_k = torch.exp((2 * torch.eq(lf_classes.view(-1, 1).double(), y.double()).double() - 1).t() * theta)
        z += (1 + M_k).prod()
    return z


def probabilities(theta, l, k, n_classes):
    prob = torch.ones(l.shape[0], n_classes)
    z = calculate_z(theta, k, n_classes)
    for y in range(n_classes):
        y_ = y * torch.ones(l.shape[0])
        prob[:, y] = un_normalized(theta, l, y_, k) / z

    return prob


def get_loss(theta, l, k, n_classes, n_lfs):
    z = calculate_z(theta, k, n_classes)
    probability = 0
    for y in range(n_classes):
        y = y * torch.ones(l.shape[0])
        probability += un_normalized(theta, l, y, k) / z
    log_probability = torch.log(probability)

    return - torch.sum(log_probability) / l.shape[0]


def get_precision_loss(theta, k, n_classes, n_lfs, a, n):
    prob = torch.ones(n_lfs, n_classes)
    z_per_lf = torch.zeros(n_lfs)
    for y in range(n_classes):
        y_ = torch.tensor(y)
        M_k = torch.exp((2 * torch.eq(k.view(-1, 1).double(), y_.double()).double() - 1).t() * theta)
        per_lf_matrix = torch.tensordot((1 + M_k).t(), torch.ones(M_k.shape).double(), 1) - torch.eye(n_lfs).double()
        prob[:, y] = per_lf_matrix.prod(0)
        z_per_lf += prob[:, y]

    prob /= z_per_lf.view(-1, 1)
    correct_prob = torch.zeros(n_lfs)
    for i in range(n_lfs):
        correct_prob[i] = prob[i, k[i]]
    loss = a * torch.log(correct_prob).double() + (1 - a) * torch.log(1 - correct_prob).double()
    loss = loss * n
    return - loss.sum()


def get_softplus_precision_loss(theta, k, n_classes, n_lfs, a, gamma):
    prob = torch.ones(n_lfs, n_classes)
    z_per_lf = torch.zeros(n_lfs)
    for y in range(n_classes):
        y_ = torch.tensor(y)
        M_k = torch.exp((2 * torch.eq(k.view(-1, 1).double(), y_.double()).double() - 1).t() * theta)
        per_lf_matrix = torch.tensordot((1 + M_k).t(), torch.ones(M_k.shape).double(), 1) - torch.eye(n_lfs).double()
        prob[:, y] = per_lf_matrix.prod(0)
        z_per_lf += prob[:, y]

    prob /= z_per_lf.view(-1, 1)
    correct_prob = torch.zeros(n_lfs)
    for i in range(n_lfs):
        correct_prob[i] = prob[i, k[i]]
    # loss = a * torch.log(correct_prob).double() + (1 - a) * torch.log(1 - correct_prob).double()
    loss = torch.nn.functional.softplus(a - correct_prob.double())
    loss = loss * gamma
    return - loss.sum()


def get_recall_loss(theta, k, n_classes, n_lfs, r, n):
    prob = torch.ones(n_lfs, n_classes)
    z_per_lf = torch.zeros(n_lfs)
    for y in range(n_classes):
        y_ = torch.tensor(y)
        M_k = torch.exp((2 * torch.eq(k.view(-1, 1).double(), y_.double()).double() - 1).t() * theta)
        per_lf_matrix = torch.tensordot((1 + M_k).t(), torch.ones(M_k.shape).double(), 1) - torch.eye(n_lfs).double()
        prob[:, y] = per_lf_matrix.prod(0)
        z_per_lf += prob[:, y]

    z = calculate_z(theta, k, n_classes)
    recall_prob = prob.sum(1).double() / z
    loss = r * torch.log(recall_prob).double() + (1 - r) * torch.log(1 - recall_prob).double()
    loss = loss * n
    return - loss.sum()