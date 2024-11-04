import torch


class ProjectionLayer(torch.nn.Module):
    def __init__(self, clip_dim, phi_dim):
        super().__init__()
        self.linear = torch.nn.Linear(clip_dim, phi_dim)

    def forward(self, x):
        return self.linear(x)