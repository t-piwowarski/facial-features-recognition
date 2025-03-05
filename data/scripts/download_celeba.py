import os
import gdown
import zipfile

# 🔹 Ścieżki do folderów
DATA_DIR = "data/celeba"
ZIP_PATH = os.path.join(DATA_DIR, "img_align_celeba.zip")
FOLDER_PATH = os.path.join(DATA_DIR, "img_align_celeba")

# 🔹 Pliki tekstowe CelebA
FILES = {
    "identity_CelebA.txt": "0B7EVK8r0v71pQy1od2FJSHpoU3M",
    "list_attr_celeba.txt": "0B7EVK8r0v71pblRyaVFSWGxPY0U",
    "list_bbox_celeba.txt": "0B7EVK8r0v71pVlg5S2ZxX01DYkk",
    "list_eval_partition.txt": "0B7EVK8r0v71pWEZsZE9oNnFqT28",
    "list_landmarks_align_celeba.txt": "0B7EVK8r0v71pdjM3NkFBSW9GZ2c"
}

# 🔹 Utwórz folder na dane
os.makedirs(DATA_DIR, exist_ok=True)

# 🔹 Pobierz CelebA (obrazy) z Google Drive
if not os.path.exists(ZIP_PATH):
    print("📥 Pobieranie obrazów CelebA...")
    gdown.download("https://drive.google.com/uc?id=0B7EVK8r0v71pZjFTYXZWM3FlRnM", ZIP_PATH, quiet=False)

# 🔹 Rozpakuj, jeśli jeszcze nie jest rozpakowane
if not os.path.exists(FOLDER_PATH):
    print("📦 Rozpakowywanie CelebA...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(DATA_DIR)
    print("✅ Obrazy CelebA rozpakowane!")

# 🔹 Pobierz pliki tekstowe
for filename, file_id in FILES.items():
    file_path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(file_path):
        print(f"📥 Pobieranie {filename}...")
        gdown.download(f"https://drive.google.com/uc?id={file_id}", file_path, quiet=False)
        print(f"✅ {filename} pobrano!")

print("🎉 Pobieranie i rozpakowywanie zakończone!")
