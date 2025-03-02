import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from src.models.gender_cnn import GenderCNNLightning

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

MALE_INDEX = 20

class CelebAGenderDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, target = self.dataset[idx]
        gender = target[MALE_INDEX].item()
        return image, gender

TRAIN_DATA_DIR = "data/celeba_train_augmented.pt"
VAL_DATA_DIR = "data/celeba_valid_augmented.pt"
TEST_DATA_DIR = "data/celeba_test_augmented.pt"

train_loader = DataLoader(CelebAGenderDataset(root=TRAIN_DATA_DIR), batch_size=32, shuffle=True)
val_loader = DataLoader(CelebAGenderDataset(root=VAL_DATA_DIR), batch_size=32, shuffle=False)
test_loader = DataLoader(CelebAGenderDataset(root=TEST_DATA_DIR), batch_size=32, shuffle=False)

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
