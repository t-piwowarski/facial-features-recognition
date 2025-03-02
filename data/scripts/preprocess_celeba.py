import os
import torch
from torchvision import transforms, datasets

DATA_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.Resize((178, 178)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def process_dataset():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    dataset = datasets.CelebA(root=DATA_DIR, split="train", target_type="attr", transform=transform, download=False)
    torch.save(dataset, os.path.join(PROCESSED_DIR, "celeba_train_augmented.pt"))

    dataset = datasets.CelebA(root=DATA_DIR, split="valid", target_type="attr", transform=transform, download=False)
    torch.save(dataset, os.path.join(PROCESSED_DIR, "celeba_valid_augmented.pt"))

    dataset = datasets.CelebA(root=DATA_DIR, split="test", target_type="attr", transform=transform, download=False)
    torch.save(dataset, os.path.join(PROCESSED_DIR, "celeba_test_augmented.pt"))

    print("Przetwarzanie zakończone!")

if __name__ == "__main__":
    process_dataset()
