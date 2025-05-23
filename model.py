import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, n_classes, in_shape=(1, 128, 256)):
        super().__init__()

        # feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),   # (16, 128, 256)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                              # (16, 64, 128)

            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # (32, 64, 128)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                              # (32, 32, 64)
        )

        # compute flattened size once
        with torch.no_grad():
            dummy = torch.zeros(1, *in_shape)
            flat_dim = self.features(dummy).numel()

        # classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(flat_dim, 256),
            nn.ReLU(inplace=True),

            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),

            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),

            nn.Dropout(0.5),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x