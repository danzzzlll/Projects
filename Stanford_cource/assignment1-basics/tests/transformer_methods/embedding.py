import torch
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        # num_embeddings: int Size of the vocabulary
        # embedding_dim: int Dimension of the embedding vectors, i.e., dmodel
        # device: torch.device | None = None Device to store the parameters on
        # dtype: torch.dtype | None = None Data type of the parameters

        embeddings = torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
        nn.init.trunc_normal_(embeddings, mean=0.0, std=1.0, a=-3, b=3)
        self.E = nn.Parameter(embeddings)

    def forward(self, token_ids: torch.Tensor):
        if self.E.device != token_ids.device:
            token_ids = token_ids.to(self.E.device)
        return self.E[token_ids]