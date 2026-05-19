import torch.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(20000, 256),
            nn.Softmax(dim=1),
            nn.Linear(256, 1),
            nn.Softmax(dim=1)
        ).to(device)

    def forward(self, x):
        return self.model(x)