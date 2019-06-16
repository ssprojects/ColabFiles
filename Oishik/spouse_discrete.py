import numpy as np
import torch
from torch import optim
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
import sys
import matplotlib.pyplot as plt
from discrete_model import get_loss, get_precision_loss, probabilities, get_softplus_precision_loss, get_recall_loss

np.set_printoptions(threshold=sys.maxsize)

n_classes = 2
n_lfs = 10
a = np.load("Data/spouse/data/train_L_S_discrete.npy")
a1 = np.load("Data/spouse/data/dev_L_S_discrete.npy")
a2 = np.load("Data/spouse/data/test_L_S_discrete.npy")

k = torch.tensor([1, 1, 1, 1, 1, 0, 1, 0, 0, 0]).long()
l = torch.tensor(a[:, 0]).long()
l_dev = torch.tensor(a1[:, 0]).long()
l_test = torch.tensor(a2[:, 0]).long()


y_true_dev = np.load("Data/spouse/data/true_labels_dev.npy")
y_true_test = np.load("Data/spouse/data/true_labels_test.npy")

a_factor = 0.9
n_factor = 1

r_factor = 0.7
nr_factor = 1

a = torch.ones(n_lfs).double() * a_factor
n = torch.ones(n_lfs).double() * n_factor

r = torch.ones(n_lfs).double() * r_factor
n_r = torch.ones(n_lfs).double() * nr_factor

a[0] = 2/3
a[1] = 7/18
a[2] = 93 / 261
a[3] = 81 / 196
a[4] = 19 / 68
a[5] = 1687 / 1741
a[6] = 1 / 8
a[7] = 317 / 332
a[8] = 210 / 211
a[9] = 21 / 25


a_factor = "dev"


r = torch.tensor(np.load("Data/spouse/recall_values.npy")).double()
r_factor = "dev"


mean = torch.tensor([1.0] * n_lfs)
std = torch.tensor([0.3] * n_lfs)

theta = torch.ones(n_lfs).double()
# theta = torch.randn(n_lfs).double()
# theta = torch.distributions.normal.Normal(mean, std).sample().double()
theta.requires_grad = True

optimizer = optim.Adam([theta], lr=0.1)

# file_type = "a_{}_n_{}".format(a_factor, n_factor)
# file_type = "softplus_a_{}_gamma_{}".format(a_factor, n_factor)
# file_type = "no_constraints"
file_type = "a_{}_n_{}_r_{}_n_{}".format(a_factor, n_factor, r_factor, nr_factor)
# file_type = "r_{}_n_{}".format(r_factor, nr_factor)
# file_type = "no_z"
theta_type = "ones"
# theta_type = "random"

f = open("Results/spouse/theta={}/Output_spouse_{}.txt".format(theta_type, file_type), "w", encoding="'utf8'")

use_precision_loss = False
use_softplus_precision_loss = False
use_recall_loss = False

use_precision_loss = True
# use_softplus_precision_loss = True
use_recall_loss = True

loss_history = []
dev_loss_history = []
test_loss_history = []
F1_history_dev = []
F1_history_test = []

for epoch in range(400):
    optimizer.zero_grad()
    loss = get_loss(theta, l, k, n_classes, n_lfs)
    precision_loss = 0
    recall_loss = 0
    if use_precision_loss:
        precision_loss = get_precision_loss(theta, k, n_classes, n_lfs, a, n)
    if use_softplus_precision_loss:
        precision_loss = get_softplus_precision_loss(theta, k, n_classes, n_lfs, a, n)
    if use_recall_loss:
        recall_loss = get_recall_loss(theta, k, n_classes, n_lfs, r, n_r)
    loss += precision_loss + recall_loss
    # print(recall_loss)
    # print(get_precision_loss(theta, k, n_classes, n_lfs, a, n))
    print(loss)
    print(theta)
    # input()

    f.write("theta for epoch {} : {}\n".format(epoch, theta))
    f.write("Train loss for epoch {} : {}\n".format(epoch, loss))

    loss_history.append(loss)
    dev_loss_history.append(get_loss(theta, l_dev, k, n_classes, n_lfs) + precision_loss)
    test_loss_history.append(get_loss(theta, l_test, k, n_classes, n_lfs) + precision_loss)

    f.write("Dev loss for epoch {} : {}\n".format(epoch, dev_loss_history[-1]))
    f.write("Test loss for epoch {} : {}\n".format(epoch, test_loss_history[-1]))
    # input()

    # y_pred = torch.argmax(probabilities(theta, l1, k, n_classes, n_lfs), 1)
    y_pred_dev = np.argmax(probabilities(theta, l_dev, k, n_classes).detach().numpy(), 1)
    y_pred_test = np.argmax(probabilities(theta, l_test, k, n_classes).detach().numpy(), 1)

    print("f1_score_dev: ", f1_score(y_true_dev, y_pred_dev, average="binary"))
    print("f1_score_test: ", f1_score(y_true_test, y_pred_test, average="binary"))

    p, r, f1_s, _ = precision_recall_fscore_support(y_true_dev, y_pred_dev, average='binary')

    f.write("Dev Precision for epoch {} : {}\n".format(epoch, p))
    f.write("Dev Recall for epoch {} : {}\n".format(epoch, r))
    f.write("Dev F1 for epoch {} : {}\n".format(epoch, f1_s))

    F1_history_dev.append(f1_s)

    p, r, f1_s, _ = precision_recall_fscore_support(y_true_test, y_pred_test, average='binary')

    f.write("Test Precision for epoch {} : {}\n".format(epoch, p))
    f.write("Test Recall for epoch {} : {}\n".format(epoch, r))
    f.write("Test F1 for epoch {} : {}\n".format(epoch, f1_s))

    f.write("\n\n")

    F1_history_test.append(f1_s)

    loss.backward()
    optimizer.step()

f.write("Max Dev F1 is for epoch {} : {}\n".format(np.argmax(F1_history_dev), np.max(F1_history_dev)))
f.write("Max Test F1 is for epoch {} : {}\n".format(np.argmax(F1_history_test), np.max(F1_history_test)))

best_loss_epoch_10 = np.argmin(dev_loss_history[:10])
best_loss_epoch_20 = np.argmin(dev_loss_history[:20])
best_loss_epoch_30 = np.argmin(dev_loss_history[:30])

f.write("Dev F1 is for epoch {} (10 epochs) : {}\n".format(best_loss_epoch_10, F1_history_dev[best_loss_epoch_10]))
f.write("Dev F1 is for epoch {} (20 epochs) : {}\n".format(best_loss_epoch_20, F1_history_dev[best_loss_epoch_20]))
f.write("Dev F1 is for epoch {} (30 epochs) : {}\n".format(best_loss_epoch_30, F1_history_dev[best_loss_epoch_30]))
f.write("Test F1 is for epoch {} (10 epochs) : {}\n".format(best_loss_epoch_10, F1_history_test[best_loss_epoch_10]))
f.write("Test F1 is for epoch {} (20 epochs) : {}\n".format(best_loss_epoch_20, F1_history_test[best_loss_epoch_20]))
f.write("Test F1 is for epoch {} (30 epochs) : {}\n".format(best_loss_epoch_30, F1_history_test[best_loss_epoch_30]))


plt.plot(loss_history, label='train_loss')
plt.plot(dev_loss_history, label='dev_loss')
plt.plot(test_loss_history, label='test_loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('loss_spouse_{}'.format(file_type))
plt.legend(loc='best', frameon=False)

plt.savefig('Results/spouse/theta={}/Images/loss/loss_spouse_{}.png'.format(theta_type, file_type))
plt.close()

plt.plot(F1_history_dev, label='F1_dev')
plt.plot(F1_history_test, label='F1_test')
plt.xlabel('Epochs')
plt.ylabel('F1_score')
plt.title('F1_spouse_{}'.format(file_type))
plt.legend(loc='best', frameon=False)
plt.savefig('Results/spouse/theta={}/Images/F1/F1_spouse_{}.png'.format(theta_type, file_type))

np.savetxt("spouse_discrete_probabilities.csv", probabilities(theta, l_dev, k, n_classes).detach().numpy(), delimiter=",")
a = probabilities(theta, l_dev, k, n_classes).detach().numpy()
c = 0
for i in range(a.shape[0]):
    if a[i][0] == a[i][1]:
        c+=1
print(c)
print(a.shape[0])