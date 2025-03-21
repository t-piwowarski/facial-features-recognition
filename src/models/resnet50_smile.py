import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchvision.models as models
from torchmetrics.classification import Accuracy

class SmileResNetLightning(pl.LightningModule):
    def __init__(self, lr=1e-4, freeze_layers=True, fine_tune_from_layer=None):
        super(SmileResNetLightning, self).__init__()
        
        # Load pre-trained ResNet50
        self.model = models.resnet50(pretrained=True)

        # Optionally freeze all layers (for transfer learning)
        if freeze_layers:
            for param in self.model.parameters():
                param.requires_grad = False

        # Optionally unfreeze specific layers for fine-tuning
        if fine_tune_from_layer:
            for name, param in self.model.named_parameters():
                if fine_tune_from_layer in name:
                    param.requires_grad = True

        # Replace the last FC layer for binary classification (smiling / not smiling)
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)

        # Loss function and accuracy metric
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task='multiclass', num_classes=2)

        # Learning rate for optimizer
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        # Separate params for fine-tuned layers and the FC head
        fc_params = set(self.model.fc.parameters())
    
        fine_tune_params = filter(
            lambda p: p.requires_grad and p not in fc_params,
            self.model.parameters()
        )

        # Use lower LR for fine-tuned layers
        params = [
            {'params': self.model.fc.parameters(), 'lr': self.lr},
            {'params': fine_tune_params, 'lr': self.lr * 0.1},
        ]
        
        optimizer = torch.optim.Adam(params)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        return [optimizer], [scheduler]
