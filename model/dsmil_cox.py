import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DSMILCox(nn.Module):
    """DSMIL dual-stream architecture adapted to Cox survival risk."""

    def __init__(self, input_size=1024, query_dim=128, node_dropout=0.0):
        super().__init__()
        if input_size <= 0:
            raise ValueError("input_size must be positive")
        if query_dim <= 0:
            raise ValueError("query_dim must be positive")
        if not 0.0 <= node_dropout < 1.0:
            raise ValueError("node_dropout must be in [0, 1)")

        self.input_size = input_size
        self.query_dim = query_dim
        self.instance_classifier = nn.Linear(input_size, 1)
        self.query = nn.Sequential(
            nn.Linear(input_size, query_dim),
            nn.ReLU(),
            nn.Linear(query_dim, query_dim),
            nn.Tanh(),
        )
        self.node_dropout = nn.Dropout(node_dropout)
        self.bag_classifier = nn.Linear(input_size, 1)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if x.dim() != 3 or x.size(-1) != self.input_size:
            raise ValueError(
                f"Expected [B, N, {self.input_size}] input, got {tuple(x.shape)}"
            )

        instance_risks = self.instance_classifier(x).squeeze(-1)
        max_instance_risk, critical_indices = instance_risks.max(dim=1)

        queries = self.query(x)
        critical_queries = queries[
            torch.arange(x.size(0), device=x.device), critical_indices
        ]
        attention_logits = torch.bmm(
            queries, critical_queries.unsqueeze(-1)
        ).squeeze(-1)
        attention_logits = attention_logits / math.sqrt(self.query_dim)
        attention = F.softmax(attention_logits, dim=1)

        values = self.node_dropout(x)
        bag_representation = torch.bmm(
            attention.unsqueeze(1), values
        ).squeeze(1)
        bag_risk = self.bag_classifier(bag_representation).squeeze(-1)

        return bag_risk, max_instance_risk, attention.unsqueeze(-1), critical_indices
