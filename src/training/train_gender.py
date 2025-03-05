import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from src.models.gender_cnn import GenderCNNLightning

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

DATA_DIR = "data"  # Musi zawierać folder "CelebA/"
MALE_INDEX = 20

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.Resize((178, 178)),
    transforms.ToTensor(),        
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

train_dataset = datasets.CelebA(root=DATA_DIR, split="train", target_type='attr', transform=transform, download=False) #000001.jpg - 162770.jpg
test_dataset = datasets.CelebA(root=DATA_DIR, split="test", target_type='attr', transform=transform, download=False) #182638.jpg - 202599.jpg
val_dataset = datasets.CelebA(root=DATA_DIR, split="valid", target_type='attr', transform=transform, download=False) #162771.jpg - 182637.jpg

class CelebAGenderDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, target = self.dataset[idx]
        gender = target[MALE_INDEX].item()
        return image, gender

train_loader = DataLoader(CelebAGenderDataset(train_dataset), batch_size=32, shuffle=True)
val_loader = DataLoader(CelebAGenderDataset(val_dataset), batch_size=32, shuffle=False)
test_loader = DataLoader(CelebAGenderDataset(test_dataset), batch_size=32, shuffle=False)

model = GenderCNNLightning()

early_stopping = EarlyStopping(monitor="val_loss", patience=5, mode="min", verbose=True)
checkpoint_callback = ModelCheckpoint(dirpath="saved_models", filename="genderCNN", save_top_k=1, monitor="val_loss", mode="min")

trainer = pl.Trainer(
    max_epochs=20,
    callbacks=[early_stopping, checkpoint_callback],
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=1
)

trainer.fit(model, train_loader, val_loader)
trainer.test(model, test_loader)
