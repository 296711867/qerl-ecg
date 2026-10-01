"""Small shared ECG classifier for continuous-stream policy comparisons."""
import torch.nn as nn


class StreamCNN(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        channels = [2, 32, 64, 128, 128]
        for a, b in zip(channels[:-1], channels[1:]):
            layers += [nn.Conv1d(a, b, 9, stride=2, padding=4),
                       nn.BatchNorm1d(b), nn.ReLU()]
        self.features = nn.Sequential(*layers, nn.AdaptiveAvgPool1d(1), nn.Flatten())
        self.head = nn.Linear(128, 3)

    def forward(self, x):
        return self.head(self.features(x))
