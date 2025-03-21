import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from src.models.resnet50_smile import SmileDetectionModel

# Select computation device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Path to CelebA dataset
DATA_DIR = "data"
SMILE_INDEX = 31  # Index of 'Smiling' attribute in CelebA

# Data augmentation and normalization
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.Resize((178, 178)),
    transforms.ToTensor(),        
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load CelebA dataset with 'Smiling' attribute
train_dataset = datasets.CelebA(root=DATA_DIR, split="train", target_type='attr', transform=transform, download=False) #000001.jpg - 162770.jpg
test_dataset = datasets.CelebA(root=DATA_DIR, split="test", target_type='attr', transform=transform, download=False) #182638.jpg - 202599.jpg
val_dataset = datasets.CelebA(root=DATA_DIR, split="valid", target_type='attr', transform=transform, download=False) #162771.jpg - 182637.jpg

# Custom dataset wrapper for extracting smile label (0 = no smile, 1 = smiling)
class CelebASmileDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, target = self.dataset[idx]
        smile = target[SMILE_INDEX].item()
        return image, smile

# Dataloaders
train_loader = DataLoader(CelebASmileDataset(train_dataset), batch_size=32, shuffle=True)
val_loader = DataLoader(CelebASmileDataset(val_dataset), batch_size=32, shuffle=False)
test_loader = DataLoader(CelebASmileDataset(test_dataset), batch_size=32, shuffle=False)

# Initialize smile detection model with ResNet50 backbone
model = SmileDetectionModel(freeze_layers=True, fine_tune_from_layer="layer4")

# Callbacks: early stopping and model checkpoint
early_stopping = EarlyStopping(monitor="val_loss", patience=5, mode="min", verbose=True)
checkpoint_callback = ModelCheckpoint(
    dirpath="saved_models",
    filename="smile_resnet",
    save_top_k=1,
    monitor="val_loss",
    mode="min"
)

# PyTorch Lightning trainer setup
trainer = pl.Trainer(
    max_epochs=20,
    callbacks=[early_stopping, checkpoint_callback],
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=1
)

# Train and test the model
trainer.fit(model, train_loader, val_loader)
trainer.test(model, test_loader)
