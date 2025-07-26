import torch
import torch.nn.functional as F

def batch_softmax(X):
    b, d = X.shape

    X_batch_block_0 = X[:, :d//2]
    X_batch_block_1 = X[:, d//2:]

    X_batch_0_max = X_batch_block_0.max(dim = 1, keepdim = True).values
    X_batch_0_sum = torch.exp(X_batch_block_0 - X_batch_0_max).sum(dim = 1, keepdim = True)

    X_batch_1_max = X_batch_block_1.max(dim = 1, keepdim = True).values
    X_batch_1_sum = torch.exp(X_batch_block_1 - X_batch_1_max).sum(dim = 1, keepdim = True)

    X_batch_1_max_update = torch.maximum(X_batch_0_max, X_batch_1_max)
    X_batch_1_sum_update = X_batch_0_sum * torch.exp(X_batch_0_max - X_batch_1_max_update) \
                        + torch.exp(X_batch_block_1 - X_batch_1_max_update).sum(dim = 1, keepdim = True) 

    X_batch_online_softmax = torch.exp(X - X_batch_1_max_update) / X_batch_1_sum_update
    return X_batch_online_softmax


def online_batch_softmax(X):
    X_blocks = torch.split(X, 2, dim=1)

    b, d = X.size()
    M_old = torch.ones((b,1)) * -100_000.0
    L_old = torch.zeros((b,1))

    for X_block in X_blocks:
        M, _ = torch.max(X_block, dim=1, keepdim=True)
        M_new = torch.maximum(M, M_old)

        L_new = L_old * torch.exp(M_old - M_new) + torch.exp(X_block - M_new).sum(dim=1, keepdim=True)
        M_old = M_new
        L_old = L_new

    X_blocks_batch = torch.exp(X - M_old) / L_old
    return X_blocks_batch