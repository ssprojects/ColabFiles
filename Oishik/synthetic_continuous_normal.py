import numpy as np
import torch
from torch import optim
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
import sys
import matplotlib.pyplot as plt
from continuous_model import *

np.set_printoptions(threshold=sys.maxsize)

n_classes = 2
n_lfs = 6

l = torch.tensor(np.load("Data/synthetic/Normal/train_l.npy")).long()
l1 = torch.tensor(np.load("Data/synthetic/Normal/test_l.npy")).long()

s = torch.tensor(np.load("Data/synthetic/Normal/train_s.npy")).double()
s1 = torch.tensor(np.load("Data/synthetic/Normal/test_s.npy")).double()

k = torch.tensor(np.load("Data/synthetic/Normal/k.npy")).long()
y_true = np.load("Data/synthetic/Normal/test_y.npy")

theta = torch.ones(n_lfs).double() * 0.5
theta.requires_grad = True

alpha = torch.ones(n_lfs).double() * 0.5
alpha.requires_grad = True

optimizer = optim.Adam([theta, alpha], lr=0.1)

f = open("Results/continuous/synthetic/Normal/Output_normal_likelihood.txt", "w", encoding="'utf8'")

for epoch in range(800):
    optimizer.zero_grad()
    # alpha = torch.max(alpha, torch.zeros(alpha.shape).double())
    # loss = em_loss(theta, alpha, l, s, k, n_classes, n_lfs)
    loss = get_loss(theta, alpha, l, s, k, n_classes, n_lfs)

    print(loss)
    print(theta)
    print(alpha)

    y_pred = np.argmax(probabilities(theta, alpha, l1, s1, k, n_classes, n_lfs).detach().numpy(), 1)
    # print(np.argmax(probabilities(theta, alpha, l1, s1, k, n_classes, n_lfs).detach().numpy(), 1))
    # print(y_true)
    # input()

    print("f1_score: ", f1_score(y_true, y_pred, average="binary"))

    p, r, f1_s, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')

    f.write("Precision for epoch {} : {}\n".format(epoch, p))
    f.write("Recall for epoch {} : {}\n".format(epoch, r))
    f.write("F1 for epoch {} : {}\n".format(epoch, f1_s))
    f.write("\n\n")

    loss.backward()
    # print(theta.grad)
    # input()
    optimizer.step()
