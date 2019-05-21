import numpy as np
import torch
from torch import optim
import pickle
import math
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
import sys
import tensorflow as tf


np.set_printoptions(threshold=sys.maxsize)


def potential(theta, l, y, k):
    # print((2 * torch.eq(k.view(-1, 1).double(), y.double()).double() - 1).t().shape)
    # input()
    return torch.tensordot((2 * torch.eq(k.view(-1, 1).double(), y.double()).double() - 1).t() * torch.abs(l.double()), theta, 1).double()


def un_normalized(theta, l, y, k):
    return torch.exp(potential(theta, l, y, k))


def calculate_z(theta, lf_classes, n_classes=2, n_lfs=2):
    z = 0
    for y in range(n_classes):
        p = 1
        y = torch.tensor(y)
        M_k = torch.exp((2 * torch.eq(lf_classes.view(-1, 1).double(), y.double()).double() - 1).t() * theta)
        z += (1 + M_k).prod()
        # print(z)
        # input()
        # for k in range(n_lfs):
        #     k1 = lf_classes[k]
        #     M_k = torch.exp((2 * torch.eq(k1.view(-1, 1).double(), y.double()).double() - 1).t() * theta[k])
        #     # print((2 * torch.eq(k.view(-1, 1).double(), y.double()).double() - 1).t() * theta[k])
        #     # # print(un_normalized(theta[k], torch.tensor(1), y, k))
        #     # input()
        #     # print(M_k.item())
        #     # input()
        #     p *= torch.tensor([1]).double() + M_k
        # z += p
    return z


def probabilities(theta, l, k, n_classes, n_lfs):
    prob = torch.ones(l.shape[0], n_classes)
    z = calculate_z(theta, k, n_classes, n_lfs)
    for y in range(n_classes):
        y_ = y * torch.ones(l.shape[0])
        prob[:, y] = un_normalized(theta, l, y_, k) / z#.double()

    return prob


def get_loss(theta, l, k, n_classes, n_lfs):
    z = calculate_z(theta, k, n_classes, n_lfs)
    # print(z)
    # input()
    probability = 0
    for y in range(n_classes):
        y = y * torch.ones(l.shape[0])
        probability += un_normalized(theta, l, y, k) / z
    log_probability = torch.log(probability)

    return (- torch.sum(log_probability)) #+ 10 * torch.norm(theta) * torch.norm(theta)


def get_precision_loss(theta, k, n_classes, n_lfs, a, n):
    prob = torch.ones(n_lfs, n_classes)
    z_per_lf = torch.zeros(n_lfs)
    for y in range(n_classes):
        y = torch.tensor(y)
        M_k = torch.exp((2 * torch.eq(k.view(-1, 1).double(), y.double()).double() - 1).t() * theta)
        per_lf_matrix = torch.tensordot((1 + M_k).t(), torch.ones(M_k.shape).double(),1) - torch.eye(n_lfs).double()
        prob[:, y] = per_lf_matrix.prod(0)
        z_per_lf += prob[:, y]
        # print((1 + M_k).t().shape)
        # print(per_lf_matrix)
        # print(per_lf_matrix.prod(0))
        # input()
    # print(prob)
    # print(z_per_lf)
    prob /= z_per_lf.view(-1, 1)
    loss = a.view(-1, 1) * torch.log(prob).double() + (1 - a).view(-1, 1)*torch.log((1 - prob)).double()
    loss = loss.sum(1) * n
    # print(prob)
    # print(loss)
    # input()
    return - loss.sum()


def load_object(filename):
    with open(filename, 'rb') as output:
        return pickle.load(output)


# n_lfs = 3
# n_classes = 2
# theta = torch.ones(n_lfs).double()
# theta.requires_grad = True
# k = torch.tensor([1, 0, 1])
# a = torch.ones(n_lfs) * 0.9
# n = torch.ones(n_lfs) * 2
# get_precision_loss(theta, k, n_classes, n_lfs, a, n)


# file1 = "lf.pkl"
# file2 = "true_labels.pkl"
#
# l = load_object(file1)
# y_true = load_object(file2)
#
# # print(l)
# # input()
#
# l = torch.tensor(l).long()
# l1 = l
#
# k = [1,2,2,2,2,3,3,3,4,4,4,4,5,5,6,7,7,7,7,8,9,10,10,11,11,12,12,12]
# k = torch.tensor(k).long()
#
# n_classes = 12
# n_lfs = 28



# test_L_S = np.load("Data/synthetic/test_synthetic_smooth.npy")
# train_L_S = np.load("Data/synthetic/train_synthetic_smooth.npy")
# dev_L_S = test_L_S
#
# gold_labels_test = test_L_S[:,0,-1]
# gold_labels_dev = gold_labels_test
# print('shape', gold_labels_test.shape)
#
# test_L_S = np.delete(test_L_S, -1, 2)#test_L_S[:][:][:-1]
# train_L_S = np.delete(train_L_S, -1, 2)#train_L_S[:][:][:-1]
# dev_L_S = np.delete(dev_L_S, -1, 2)#dev_L_S[:][:][:-1]
#
# print(test_L_S.shape,train_L_S.shape)
# n_classes=2
# print(n_classes)
# n_lfs = test_L_S.shape[2]
# print(n_lfs)
#
# # test_L_S[test_L_S[:, 0, 9] == LF_l[9],0,9]
# # LF_names= [lf.__name__ for lf in LFs]
# # print(train_L_S[0,:,:10])
#
#
#
#
# print(train_L_S[1])
# for i,x in enumerate(train_L_S):
#     train_L_S[i][0][0] = 1 if x[1][0] > 0.5 else 0
#     train_L_S[i][0][1] = -1 if x[1][1] > 0.5 else 0
#
# # print(test_L_S[:, 0, :])
#
# l = torch.tensor(train_L_S[:, 0, :])
# l1 = torch.tensor(test_L_S[:, 0, :])
#
#
# y_true = gold_labels_test
# k = torch.tensor([1, 0]).long()
# print(y_true)

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

a = torch.ones(n_lfs).double() * 0.7
n = torch.ones(n_lfs).double() * 10000

# n_lfs = 2
# n_classes = 2

# l = torch.tensor([[1, 0], [1, -1], [0, -1], [1, -1]]).long()

theta = torch.ones(n_lfs).double()
theta.requires_grad = True
#
# print(probabilities(theta, l, k, n_classes, n_lfs))
# print(get_loss(theta, l, k, n_classes, n_lfs))
# input()

optimizer = optim.Adam([theta], lr=0.1)

f = open("Output_own.txt", "w", encoding="'utf8'")

for epoch in range(100):
    optimizer.zero_grad()
    loss = get_loss(theta, l, k, n_classes, n_lfs)
    loss += get_precision_loss(theta, k, n_classes, n_lfs, a, n)
    # print(get_precision_loss(theta, k, n_classes, n_lfs, a, n))
    # loss.backward()
    # print(theta.grad)
    print(loss)
    print(theta)
    # print(probabilities(theta, l1, k, n_classes, n_lfs))
    # input()

    # y_pred = torch.argmax(probabilities(theta, l1, k, n_classes, n_lfs), 1)
    y_pred = np.argmax(probabilities(theta, l1, k, n_classes, n_lfs).detach().numpy(), 1)
    # print(y_pred)
    # input()
    # print(l1.shape)
    # print(y_pred.shape)
    # print(y_true)
    print("f1_score: ", f1_score(y_true, y_pred, average="binary"))

    p, r, f1_s, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')

    f.write("Precision for epoch {} : {}\n".format(epoch, p))
    f.write("Recall for epoch {} : {}\n".format(epoch, r))
    f.write("F1 for epoch {} : {}\n".format(epoch, f1_s))
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
