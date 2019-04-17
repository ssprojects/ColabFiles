import tensorflow as tf
import numpy as np
import math
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import sys

np.set_printoptions(threshold=sys.maxsize)


def gen_data(n_example, n_classes, n_lf_per_class):
    l = []
    y = []
    k = []
    for m in range(n_example):
        c = np.random.randint(n_classes)
        y.append(c)

        l_temp = np.zeros(np.sum(n_lf_per_class))

        s = 0
        for i in range(n_classes):
            for j in range(n_lf_per_class[i]):
                if m == 0:
                    k.append(i)
                c1 = np.random.uniform(0, 1)
                if i == c and c1 > 0.3:
                    l_temp[s] = 1
                elif c1 > 0.8:
                    l_temp[s] = 1
                s+=1
        l.append(l_temp)
    return np.array(l), k, y




a = np.load("Data/spouse/data/train_L_S_discrete.npy")

m = 1000
n_classes = 2
n_lfs = 10
n_lf_per_class = [5, 5]

k = tf.convert_to_tensor([1, 1, 1, 1, 1, -1, -1, -1, -1, -1], dtype=tf.float64)
l = tf.convert_to_tensor(a[:, 0], dtype=tf.float64)
y_ = tf.convert_to_tensor([1, -1], dtype=tf.float64)

# l, k, _ = gen_data(m, n_classes, n_lf_per_class)


# k = tf.convert_to_tensor(k, dtype=tf.float64)
# l = tf.convert_to_tensor(l)

m = l.shape[0]
print(m)

a = tf.convert_to_tensor([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=tf.float64)
n = tf.convert_to_tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=tf.float64)

def equals(k, y):
    r = tf.cast(tf.equal(k, y), tf.float64)

    # sess = tf.Session()
    # sess.run(tf.global_variables_initializer())
    # print(sess.run(tf.equal(k, y)))
    # print(sess.run(y))
    # print(sess.run(k))
    # print(sess.run(r))
    # input()
    r = 2 * r - tf.ones_like(r)

    return r

def potential(theta, k, y, l):
    # print(l, k, y, theta)
    return tf.abs(l) * equals(k, y) * theta

def calculate_z(theta, k):
    z = tf.map_fn(lambda y: tf.exp(tf.reduce_sum(potential(theta, k, y, tf.ones_like(k)))) + 1, y_)
    z = tf.reduce_sum(z)
    return z


def calculate_z_1(theta, k):
    z = tf.map_fn(lambda y: tf.exp(tf.reduce_sum(potential(theta, k, y, tf.ones_like(k)))), y_)
    z = tf.reduce_sum(z)
    return z


def model(theta, k, l):
    # sess = tf.Session()
    # sess.run(tf.global_variables_initializer())
    # print(potential(theta, k, y_[0], l), l)
    # input()
    # print(sess.run(tf.reduce_sum(potential(theta, k, y_[0], l), axis=1)))
    # input()
    return tf.reduce_sum(tf.map_fn(lambda y: tf.exp(tf.reduce_sum(potential(theta, k, y, l), axis=1)) / calculate_z(theta, k), y_), axis=1)

def model1(theta, k, l):
    # sess = tf.Session()
    # sess.run(tf.global_variables_initializer())
    # print(sess.run(tf.reduce_sum(potential(theta, k, k, l), axis=1)))
    # input()
    return tf.exp(tf.reduce_sum(potential(theta, k, k, l), axis=1)) / calculate_z_1(theta, k)

def get_loss(theta, k, l, m):
    lo = 0
    # for i in range(m):
    #     lo += tf.log(model(theta, k, l[i]))
    # i=0
    # while (i + 1)*1000 < m:
    #     lo += tf.log(model(theta, k, l[i*1000:(i+1)*1000])) + tf.norm(thetas)
    #     i+=1
    lo = tf.log(model(theta, k, l))
    return 0 - tf.reduce_sum(lo) - 0.1*tf.norm(thetas)


def get_loss_1(theta, k, l, m, a, n):
    l1 = 0
    for i in range(n_lfs):
        l1 += n[i] * (a[i]*tf.log(model1(theta, k, l)) + (1-a[i])*tf.log(1 - model1(theta, k, l)))

    lo = l1
    # for i in range(m):
    #     lo += tf.log(model(theta, k, l[i]))
    lo += tf.log(model(theta, k, l)) + tf.norm(thetas)
    return 0 - lo

def probability(theta, k, l):
    return tf.map_fn(lambda y: tf.exp(tf.reduce_sum(potential(theta, k, y, l), axis=1)) / calculate_z(theta, k), y_)


thetas = tf.squeeze(tf.ones((1, n_lfs), dtype=tf.float64))
thetas = tf.Variable(thetas)
optimizer = tf.train.AdamOptimizer(learning_rate=0.1)

loss = get_loss(thetas, k, l, m)

g = tf.gradients(loss, thetas)
opt = optimizer.minimize(loss=loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
# print(sess.run(tf.exp(tf.reduce_sum(potential(thetas, k, y_[0], l), axis=1))))
# input()
print(tf.norm(thetas).eval(session=sess))
f = open("Output.txt", "w", encoding="'utf8'")

a1 = np.load("Data/spouse/data/test_L_S_discrete.npy")
l1 = tf.convert_to_tensor(a1[:, 0], dtype=tf.float64)
y_true = np.load("Data/spouse/data/true_labels_test.npy")

# l1, _, y_true = gen_data(1000, 2, n_lf_per_class)
# k = tf.convert_to_tensor(k, dtype=tf.float64)
# l1 = tf.convert_to_tensor(l1)
# y_true = tf.convert_to_tensor(y_true, dtype=tf.float64)

for i in range(100):
    #print(sess.run(loss))
    # print(sess.run(g))
    # print(sess.run(thetas))
    y_pred = tf.argmax(probability(thetas, k, l1), axis=0)
    # print(probability(thetas, k, l1).eval(session=sess))
    # input()
    print("Precision for epoch {} : {}".format(i, average_precision_score(y_true, y_pred.eval(session=sess))))
    print("Recall for epoch {} : {}".format(i, recall_score(y_true, y_pred.eval(session=sess))))
    print("F1 for epoch {} : {}".format(i, f1_score(y_true, y_pred.eval(session=sess))))
    f.write("Precision for epoch {} : {}\n".format(i, average_precision_score(y_true, y_pred.eval(session=sess))))
    f.write("Recall for epoch {} : {}\n".format(i, recall_score(y_true, y_pred.eval(session=sess))))
    f.write("F1 for epoch {} : {}\n".format(i, f1_score(y_true, y_pred.eval(session=sess))))
    f.write("\n\n")
    sess.run(opt)
    loss = get_loss(thetas, k, l, m)
    g = tf.gradients(loss, thetas)

f.close()

a = np.load("Data/spouse/data/dev_L_S_discrete.npy")
l = tf.convert_to_tensor(a[:, 0], dtype=tf.float64)
y_true = np.load("Data/spouse/data/true_labels_dev.npy")

# print(sess.run(probability(thetas, k, l)))
# thetas = tf.squeeze(tf.ones((1, n_lfs), dtype=tf.float64))
y_pred = tf.argmax(probability(thetas, k, l), axis=0)
print(average_precision_score(y_true, y_pred.eval(session=sess)))
print(recall_score(y_true, y_pred.eval(session=sess)))
print(f1_score(y_true, y_pred.eval(session=sess)))
# print(sess.run(y_pred))
# a = np.load("Data/spouse/data/true_labels_dev.npy")
#
# print(a)