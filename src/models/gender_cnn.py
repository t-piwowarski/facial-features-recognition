import torch
from torch import nn
import pytorch_lightning as pl
from torchmetrics.classification import Accuracy

class GenderCNNLightning(pl.LightningModule):
    def __init__(self):
        super(GenderCNNLightning, self).__init__()

        # Feature extractor – 4 convolutional blocks with ReLU, BatchNorm and MaxPool
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

        # Dynamically determine the flattened size after conv layers
        with torch.no_grad():
            sample_input = torch.zeros((1, 3, 178, 178))
            sample_output = self.model(sample_input)
            self.flatten_size = sample_output.numel()

        # Fully connected layers for classification
        self.fc = nn.Sequential(
            nn.Linear(self.flatten_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)  # Binary classification (Male / Female)
        )

        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task='binary')

    def forward(self, x):
        # Forward pass through conv + fc
        x = self.model(x)
        x = self.flatten(x)
        return self.fc(x)
    
    def training_step(self, batch, batch_idx):
        # Training logic for one batch
        images, labels = batch
        outputs = self(images)
        loss = self.loss_fn(outputs, labels)

        acc = self.accuracy(outputs.argmax(dim=1), labels)

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # Validation logic for one batch
        images, labels = batch
        outputs = self(images)
        loss = self.loss_fn(outputs, labels)

        acc = self.accuracy(outputs.argmax(dim=1), labels)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        # Optimizer + learning rate scheduler
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        return [optimizer], [scheduler]
