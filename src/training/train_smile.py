import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from src.models.resnet50_smile import SmileDetectionModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

SMILE_INDEX = 31

class CelebASmileDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, target = self.dataset[idx]
        smile = target[SMILE_INDEX].item()
        return image, smile

TRAIN_DATA_DIR = "data/celeba_train_augmented.pt"
VAL_DATA_DIR = "data/celeba_valid_augmented.pt"
TEST_DATA_DIR = "data/celeba_test_augmented.pt"

train_loader = DataLoader(CelebASmileDataset(root=TRAIN_DATA_DIR), batch_size=32, shuffle=True)
val_loader = DataLoader(CelebASmileDataset(root=VAL_DATA_DIR), batch_size=32, shuffle=False)
test_loader = DataLoader(CelebASmileDataset(root=TEST_DATA_DIR), batch_size=32, shuffle=False)

# 🎯 Tworzenie modelu
model = SmileDetectionModel(freeze_layers=True, fine_tune_from_layer="layer4")

# 📌 Callbacki
early_stopping = EarlyStopping(monitor="val_loss", patience=5, mode="min", verbose=True)
checkpoint_callback = ModelCheckpoint(dirpath="saved_models", filename="smile_resnet", save_top_k=1, monitor="val_loss", mode="min")

trainer = pl.Trainer(
    max_epochs=20,
    callbacks=[early_stopping, checkpoint_callback],
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=1
)

# 🏋️‍♂️ Trening
trainer.fit(model, train_loader, val_loader)

# 💾 Testowanie
trainer.test(model, test_loader)
