import numpy as np
import torch
from scipy.stats import beta
import sys
from torch.distributions.beta import Beta
from torch.distributions.half_normal import HalfNormal

np.set_printoptions(threshold=sys.maxsize)


def probability_y(pi):
    return pi / pi.sum()


def probabilty_s_given_y(theta, s, y, l, k, ratio_agreement=0.95, model=1):
    if model == 1:
        eq = torch.eq(k.view(-1, 1).long(), y.long()).double().t()
        r = ratio_agreement * eq.squeeze() + (1 - ratio_agreement) * (1 - eq.squeeze())
        eq = torch.stack([eq, 1 - eq]).squeeze().t()
        params = (theta * eq).sum(1)
        probability = 1
        for i in range(k.shape[0]):
            m = Beta(r[i] * params[i] / (r[i] + 1), params[i] / (r[i] + 1))
            probability *= torch.exp(m.log_prob(s[:, i].double(), )) * l[:, i].double() + (1 - l[:, i]).double()
    elif model == 2:
        eq = torch.eq(k.view(-1, 1).long(), y.long()).double().t()
        eq = torch.stack([eq, 1 - eq]).squeeze().t()
        params = (theta * eq).sum(1)
        probability = 1
        for i in range(k.shape[0]):
            m = HalfNormal(params[i])
            probability *= ((1 - torch.exp(m.log_prob(s[:, i].double()))) * eq[i, 0] + (torch.exp(m.log_prob(s[:, i].double())))
                            * (1 - eq[i, 0])) * l[:, i].double() + (1 - l[:, i]).double()
    return probability


def probabilities(theta, pi, s, l, k, n_classes, model=1):
    probability = torch.zeros((s.shape[0], n_classes))
    for y in range(n_classes):
        probability[:, y] = probability_y(pi)[y] * probabilty_s_given_y(theta[:, y], s, torch.tensor(y), l, k, model=model)
    return probability


def expectation(theta, pi, s, l, k, n_classes, model=1):
    q = probabilities(theta, pi, s, l, k, n_classes, model)
    q = q / q.sum(1).view(-1, 1)
    return q.detach()


def maximization(theta, pi, s, l, k, n_classes, q, model=1):
    p = probabilities(theta, pi, s, l, k, n_classes, model)
    return (q * torch.log(p)).sum()


# theta = torch.tensor([[[1, 2], [1, 3], [1, 4]], [[2, 1], [2, 3], [2, 4]], [[3, 1], [3, 2], [3, 4]]]).double()
# pi = torch.tensor([1., 2., 3.])
#
# k = torch.tensor([0, 1, 2])
# y = torch.tensor([1])
#
# l = torch.tensor([[1, 0, 1], [0, 1, 1]])
# s = torch.tensor([[0.7, 0.4, 0.9], [0.2, 0.6, 0.8]])
# print(probability_y(pi))
# print(probabilty_s_given_y(theta[:, y.item()], s, y, l, k))
# print(maximization(theta, pi, s, l, k, 3, expectation(theta, pi, s, l, k, 3)))