import torch
import torch.nn as nn


def online_softmax_1d(X):
    x_pre = X[:-1]
    x_pre_max = x_pre.max()
    x_pre_sum = torch.exp(x_pre - x_pre_max).sum()

    x_cur_max = torch.max(x_pre_max, X[-1])
    x_cur_sum = x_pre_sum * torch.exp(x_pre_max - x_cur_max) + torch.exp(X[-1] - x_cur_max)
    soft = torch.exp(X - x_cur_max) / x_cur_sum 
    return soft