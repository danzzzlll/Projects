import torch

def block_softmax(X):
    X_block = torch.split(X, split_size_or_sections = 4, dim = 0) 

    X_block_0_max = X_block[0].max()
    X_block_0_sum = torch.exp(X_block[0] - X_block_0_max).sum()

    X_block_1_max = X_block[1].max()
    X_block_1_sum = torch.exp(X_block[1] - X_block_1_max).sum()

    X_max_global = torch.max(X_block_0_max, X_block_1_max) 
    L_global = (X_block_0_sum * torch.exp(X_block_0_max - X_max_global) \
                + X_block_1_sum * torch.exp(X_block_1_max - X_max_global)) # block sum

    X_block_online_softmax_parallel = torch.exp(X - X_max_global) / L_global

    return X_block_online_softmax_parallel


def online_block_softmax(X):
    X_block = torch.split(X, split_size_or_sections = 2, dim = 0) 

    M_old = torch.tensor([-10_000.])
    L_old = torch.tensor([0.])

    for block in X_block:
        M = torch.max(block)
        M_new = torch.max(M, M_old) 

        L_new = L_old * torch.exp(M_old - M_new) + torch.exp(block - M).sum() * torch.exp(M - M_new) 

        M_old = M_new
        L_old = L_new

    soft = torch.exp(X - M_old) / L_old
    return soft