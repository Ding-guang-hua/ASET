import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import os

def compute_distinct(adj_matrix): 

    rows, cols = adj_matrix.nonzero()
    e1_list = torch.from_numpy(rows)
    e2_list = torch.from_numpy(cols)

    def get_rows(indices):
        return torch.from_numpy(adj_matrix[indices.numpy()].toarray()).float()

    emb1 = get_rows(e1_list)
    emb2 = get_rows(e2_list)


    cos_sims = torch.cosine_similarity(emb1, emb2, dim=1)
    avg_cos = torch.mean(cos_sims)

    distinct_score = 1.0 / (1.0 + avg_cos)
    return distinct_score.item() if isinstance(distinct_score, torch.Tensor) else distinct_score

def compute_distinct_drug(adj_matrix):  


    rows, cols = adj_matrix.nonzero()
    total = len(rows)

    batch_size = 2048
    all_cos = []

    for i in range(0, total, batch_size):
        end = min(i + batch_size, total)


        batch_e1 = rows[i:end]
        batch_e2 = cols[i:end]

        emb1 = torch.from_numpy(adj_matrix[batch_e1].toarray()).float()
        emb2 = torch.from_numpy(adj_matrix[batch_e2].toarray()).float()

        cos_sims = torch.cosine_similarity(emb1, emb2, dim=1)
        all_cos.append(cos_sims)


    all_cos = torch.cat(all_cos)
    avg_cos = torch.mean(all_cos)
    distinct_score = 1.0 / (1.0 + avg_cos)

    return distinct_score.item()

def compute_sim(adj_i, adj_j):

    rows_i, _ = adj_i.nonzero()
    rows_j, _ = adj_j.nonzero()

    set_i = set(rows_i)
    set_j = set(rows_j)

    intersection = len(set_i & set_j)
    union = len(set_i | set_j)

    return intersection / union if union != 0 else 0.0

