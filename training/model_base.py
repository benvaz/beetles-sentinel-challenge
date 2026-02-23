import torch.nn as nn


class DINO_DeepRegressor(nn.Module):
    """DinoV2 frozen backbone + conv spatial pooling + MLP regressor."""
    def __init__(self, dino, hid=512, shrink=4, n_out=3):
        super().__init__()
        self.dino = dino
        self.tokens_to_linear = nn.Sequential(
            nn.Conv2d(768, 768, kernel_size=5, stride=1, padding=0),   # Bx768x16x16 -> Bx768x12x12
            nn.ReLU(),
            nn.Conv2d(768, 1024, kernel_size=12, stride=1, padding=0), # -> Bx1024x1x1
            nn.ReLU(),
        )
        self.regressor = nn.Sequential(
            nn.Linear(1024, hid), nn.GELU(),
            nn.Linear(hid, hid // shrink), nn.GELU(),
            nn.Linear(hid // shrink, hid // shrink**2), nn.GELU(),
            nn.Linear(hid // shrink**2, n_out),
        )

    def forward(self, x):
        tok = self.dino(x)[0][:, 1:]
        # Bx256x768 -> Bx768x16x16 (preserve spatial layout)
        t = tok.transpose(1, 2).unflatten(dim=2, sizes=(16, 16))
        return self.regressor(self.tokens_to_linear(t).squeeze())
