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
            M_k = torch.exp((2 * torch.eq(k.view(-1, 1).double(), y.double()).double() - 1).t() * theta[k])
            p *= 1 + M_k.item()
        z += p
    return z


def probabilities(theta, l, k, n_classes, n_lfs):
    prob = torch.ones(l.shape[0], n_classes)
    z = calculate_z(theta, n_classes, n_lfs)
    for y in range(n_classes):
        y_ = y * torch.ones(l.shape[0])
        prob[:, y] = un_normalized(theta, l, y_, k) / z#.double()

    return prob


def get_loss(theta, l, k, n_classes, n_lfs):
    z = calculate_z(theta, n_classes, n_lfs)
    probability = 0
    for y in range(n_classes):
        y = y * torch.ones(l.shape[0])
        probability += un_normalized(theta, l, y, k) / z
    log_probability = torch.log(probability)

    return (- torch.sum(log_probability)) + 10 * torch.norm(theta) * torch.norm(theta)



test_L_S = np.load("Data/synthetic/test_synthetic_smooth.npy")
train_L_S = np.load("Data/synthetic/train_synthetic_smooth.npy")
dev_L_S = test_L_S

gold_labels_test = test_L_S[:,0,-1]
gold_labels_dev = gold_labels_test
print('shape', gold_labels_test.shape)

test_L_S = np.delete(test_L_S, -1, 2)#test_L_S[:][:][:-1]
train_L_S = np.delete(train_L_S, -1, 2)#train_L_S[:][:][:-1]
dev_L_S = np.delete(dev_L_S, -1, 2)#dev_L_S[:][:][:-1]

print(test_L_S.shape,train_L_S.shape)
n_classes=2
print(n_classes)
n_lfs = test_L_S.shape[2]
print(n_lfs)

for i,x in enumerate(train_L_S):
    train_L_S[i][0][0] = 1 if x[1][0] > 0.5 else 0
    train_L_S[i][0][1] = -1 if x[1][1] > 0.5 else 0

# print(test_L_S[:, 0, :])

l = torch.tensor(train_L_S[:, 0, :])
l1 = torch.tensor(test_L_S[:, 0, :])


y_true = gold_labels_test
k = torch.tensor([1, 0]).long()
# print(y_true)

# n_classes = 2
# n_lfs = 10
# a = np.load("Data/spouse/data/train_L_S_discrete.npy")
# a1 = np.load("Data/spouse/data/test_L_S_discrete.npy")
#
# k = torch.tensor([1, 1, 1, 1, 1, 0, 0, 0, 0, 0]).long()
# l = torch.tensor(a[:, 0]).long()
# l1 = torch.tensor(a1[:, 0]).long()
#
# print(l.shape)
# print(l1.shape)
# m = l.shape[0]
# print(m)
#
# y_true = np.load("Data/spouse/data/true_labels_test.npy")

# n_lfs = 2
# n_classes = 2
#
# l = torch.tensor([[1, 0], [1, -1], [0, -1], [1, -1]]).long()
#
theta = torch.ones(n_lfs).double()
theta.requires_grad = True


optimizer = optim.Adam([theta], lr=0.1)

f = open("Output_test_ones_synthetic_l_10.txt", "w", encoding="'utf8'")

for epoch in range(100):
    optimizer.zero_grad()
    loss = get_loss(theta, l, k, n_classes, n_lfs)
    print(loss)
    print(theta)

    y_pred = torch.argmax(probabilities(theta, l1, k, n_classes, n_lfs), 1)
    print("f1_score: ", f1_score(y_true, y_pred))

    f.write("Precision for epoch {} : {}\n".format(epoch, average_precision_score(y_true, y_pred)))
    f.write("Recall for epoch {} : {}\n".format(epoch, recall_score(y_true, y_pred)))
    f.write("F1 for epoch {} : {}\n".format(epoch, f1_score(y_true, y_pred)))
    f.write("\n\n")

    loss.backward()
    optimizer.step()

