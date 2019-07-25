import numpy as np
import torch
from torch import optim
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
import sys
import matplotlib.pyplot as plt
from bayesian_continuous_model import *

np.set_printoptions(threshold=sys.maxsize)

n_classes = 2
n_lfs = 6

l = torch.tensor(np.load("Data/synthetic/Normal/train_l.npy")).long()
l_test = torch.tensor(np.load("Data/synthetic/Normal/test_l.npy")).long()

s = torch.tensor(np.load("Data/synthetic/Normal/train_s.npy")).double()
s_test = torch.tensor(np.load("Data/synthetic/Normal/test_s.npy")).double()

k = torch.tensor(np.load("Data/synthetic/Normal/k.npy")).long()
y_true = np.load("Data/synthetic/Normal/test_y.npy")

y = torch.tensor(np.load("Data/synthetic/Beta/train_y.npy")).double()

for i in range(s.shape[0]):
    for j in range(s.shape[1]):
        if s[i, j] == 1:
            s[i, j] = 0.9
        if s[i, j] == 0:
            s[i, j] = 0.1

for i in range(s_test.shape[0]):
    for j in range(s_test.shape[1]):
        if s_test[i, j] == 1:
            s_test[i, j] = 0.9
        if s_test[i, j] == 0:
            s_test[i, j] = 0.1

theta = torch.ones((n_lfs, n_classes, 2)).double()
theta.requires_grad = True

pi = torch.ones(n_classes).double()
pi.requires_grad = True

optimizer = optim.Adam([theta, pi], lr=0.1)

f = open("Results/continuous/synthetic/Normal_bayesian/Output_beta_em.txt", "w", encoding="'utf8'")

loss_history = []
test_loss_history = []
F1_history_dev = []
F1_history_test = []

model = 2
supervised = False

# print(probability_y(pi))
# print(probabilities(theta, pi, s_test, l_test, k, n_classes).detach().numpy())
# input()

for epoch in range(100):
    optimizer.zero_grad()
    if supervised:
        q = torch.stack([y, 1 - y]).t().float()

        loss = - maximization(theta, pi, s, l, k, n_classes, q, model=model)

        print(loss)
        print(theta)
        print(pi)

        loss_history.append(loss)
        q_test = torch.stack([torch.tensor(y_true), torch.tensor(1 - y_true)]).t().float()
        test_loss_history.append(- maximization(theta, pi, s_test, l_test, k, n_classes, q_test, model=model))

        y_pred = np.argmax(probabilities(theta, pi, s_test, l_test, k, n_classes, model=model).detach().numpy(), 1)

        print("f1_score: ", f1_score(y_true, y_pred, average="binary"))

        p, r, f1_s, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')

        f.write("Precision for epoch {} : {}\n".format(epoch, p))
        f.write("Recall for epoch {} : {}\n".format(epoch, r))
        f.write("F1 for epoch {} : {}\n".format(epoch, f1_s))
        f.write("\n\n")

        F1_history_test.append(f1_s)

        loss.backward()
        optimizer.step()
    else:
        q = expectation(theta, pi, s, l, k, n_classes, model=model)
        old_loss = 0
        for i in range(800):
            loss = - maximization(theta, pi, s, l, k, n_classes, q, model=model)

            print(loss)
            print(theta)
            print(pi)

            loss.backward()
            # print(theta.grad)
            t_grad = np.abs(theta.grad.max().item())
            p_grad = np.abs(pi.grad.max().item())
            # input()
            optimizer.step()
            # print("Grad")
            # print(t_grad)
            # print(p_grad)

            if np.abs(loss.item() - old_loss) < 0.001:
                break

            if t_grad < 0.001 and p_grad < 0.001:
                break

            old_loss = loss.item()
        loss_history.append(loss)
        q_test = - expectation(theta, pi, s_test, l_test, k, n_classes, model=model)
        test_loss_history.append(- maximization(theta, pi, s_test, l_test, k, n_classes, q_test, model=model))

        y_pred = np.argmax(probabilities(theta, pi, s_test, l_test, k, n_classes, model=model).detach().numpy(), 1)

        print("f1_score: ", f1_score(y_true, y_pred, average="binary"))

        p, r, f1_s, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')

        f.write("Precision for epoch {} : {}\n".format(epoch, p))
        f.write("Recall for epoch {} : {}\n".format(epoch, r))
        f.write("F1 for epoch {} : {}\n".format(epoch, f1_s))
        f.write("\n\n")

        F1_history_test.append(f1_s)


plt.plot(loss_history, label='train_loss')
plt.plot(test_loss_history, label='test_loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('loss_beta_em')
plt.legend(loc='best', frameon=False)

plt.savefig('Results/continuous/synthetic/Normal_bayesian/loss/loss_normal_em.png')
plt.close()


plt.plot(F1_history_test, label='F1_test')
plt.xlabel('Epochs')
plt.ylabel('F1_score')
plt.title('F1_beta_em')
plt.legend(loc='best', frameon=False)
plt.savefig('Results/continuous/synthetic/Normal_bayesian/F1/F1_normal_em.png')