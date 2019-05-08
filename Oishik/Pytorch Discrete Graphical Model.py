import numpy as np
import torch
from torch import optim
import math
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import sys

np.set_printoptions(threshold=sys.maxsize)


def potential(theta, l, y, k):
    # print((2 * torch.eq(k.view(-1, 1).double(), y.double()).double() - 1).t().shape)
    # input()
    return torch.tensordot((2 * torch.eq(k.view(-1, 1).double(), y.double()).double() - 1).t() * torch.abs(l.double()), theta, 1).double()


def un_normalized(theta, l, y, k):
    return torch.exp(potential(theta, l, y, k))


def calculate_z(theta, n_classes=2, n_lfs=2):
    z = 0
    for y in range(n_classes):
        p = 1
        y = torch.tensor(y)
        for k in range(n_lfs):
            k = torch.tensor(k)
            p *= 1 + un_normalized(theta, torch.tensor(1), y, k)
        z += p
    return z


def probabilities(theta, l, k, n_classes, n_lfs):
    prob = torch.ones(l.shape[0], n_classes)
    z = calculate_z(theta, n_classes, n_lfs)
    for y in range(n_classes):
        y_ = y * torch.ones(l.shape[0])
        prob[:, y] = un_normalized(theta, l, y_, k) / z.double()

    return prob


def get_loss(theta, l, k, n_classes, n_lfs):
    z = calculate_z(theta, n_classes, n_lfs)
    probability = 0
    for y in range(n_classes):
        y = y * torch.ones(l.shape[0])
        probability += un_normalized(theta, l, y, k) / z
    log_probability = torch.log(probability)

    return (- torch.sum(log_probability)) #+ 0.1 * torch.norm(theta)


n_classes = 2
n_lfs = 10
a = np.load("Data/spouse/data/train_L_S_discrete.npy")
a1 = np.load("Data/spouse/data/dev_L_S_discrete.npy")

k = torch.tensor([1, 1, 1, 1, 1, 0, 0, 0, 0, 0]).long()
l = torch.tensor(a[:, 0]).long()
l1 = torch.tensor(a1[:, 0]).long()

print(l.shape)
print(l1.shape)
m = l.shape[0]
print(m)

y_true = np.load("Data/spouse/data/true_labels_dev.npy")

theta = torch.rand(n_lfs).double()
theta.requires_grad = True

optimizer = optim.Adam([theta], lr=0.1)

f = open("Output_dev.txt", "w", encoding="'utf8'")

for epoch in range(100):
    optimizer.zero_grad()
    loss = get_loss(theta, l, k, n_classes, n_lfs)
    print(loss)
    print(theta)

    y_pred = torch.argmax(probabilities(theta, l1, k, n_classes, n_lfs), 1)
    # print(l1.shape)
    # print(y_pred.shape)
    # print(y_true)
    print("f1_score: ", f1_score(y_true, y_pred))

    f.write("Precision for epoch {} : {}\n".format(epoch, average_precision_score(y_true, y_pred)))
    f.write("Recall for epoch {} : {}\n".format(epoch, recall_score(y_true, y_pred)))
    f.write("F1 for epoch {} : {}\n".format(epoch, f1_score(y_true, y_pred)))
    f.write("\n\n")

    loss.backward()
    optimizer.step()

# y_pred = torch.argmax(probabilities(theta, l, k, n_classes, n_lfs), 1)
# print(f1_score(y_true, y_pred))





# a = torch.tensor([1,2])
# b = torch.tensor([2, 2, 1])
# # print((2 * torch.eq(a.view(-1,1), b).long() - 1).t())
# theta = torch.tensor([1, 2])
# l = torch.tensor([[1, 0], [0, -1], [1, 1]])
# print(potential(theta, l, b, a))
# print(get_loss(theta, l, a, 2, 2))

