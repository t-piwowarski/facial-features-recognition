import torch
from torch import nn
import pytorch_lightning as pl
from torchmetrics.classification import Accuracy

class GenderCNNLightning(pl.LightningModule):
    def __init__(self):
        super(GenderCNNLightning, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.flatten = nn.Flatten()

        # Automatic calculation of the input size for a fully connected layer
        with torch.no_grad():
            sample_input = torch.zeros((1, 3, 178, 178))
            sample_output = self.model(sample_input)
            self.flatten_size = sample_output.numel()

        self.fc = nn.Sequential(
            nn.Linear(self.flatten_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)  # Output for 2 classes (men/women)
        )

        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task='binary')

    def forward(self, x):
        x = self.model(x)
        x = self.flatten(x)
        return self.fc(x)
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=0.001, weight_decay=1e-4)
