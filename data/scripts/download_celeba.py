import os
import gdown
import zipfile

# Paths for dataset storage
DATA_DIR = "data/celeba"
ZIP_PATH = os.path.join(DATA_DIR, "img_align_celeba.zip")
FOLDER_PATH = os.path.join(DATA_DIR, "img_align_celeba")

# Google Drive file IDs for CelebA metadata files
FILES = {
    "identity_CelebA.txt": "0B7EVK8r0v71pQy1od2FJSHpoU3M",
    "list_attr_celeba.txt": "0B7EVK8r0v71pblRyaVFSWGxPY0U",
    "list_bbox_celeba.txt": "0B7EVK8r0v71pVlg5S2ZxX01DYkk",
    "list_eval_partition.txt": "0B7EVK8r0v71pWEZsZE9oNnFqT28",
    "list_landmarks_align_celeba.txt": "0B7EVK8r0v71pdjM3NkFBSW9GZ2c"
}

# Create the dataset directory if it doesn't exist
os.makedirs(DATA_DIR, exist_ok=True)

# Download the image archive if it's not already present
if not os.path.exists(ZIP_PATH):
    print("Downloading CelebA image archive...")
    gdown.download("https://drive.google.com/uc?id=0B7EVK8r0v71pZjFTYXZWM3FlRnM", ZIP_PATH, quiet=False)

# Extract the archive if the image folder doesn't exist
if not os.path.exists(FOLDER_PATH):
    print("Extracting CelebA images...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(DATA_DIR)
    print("Extraction complete.")

# Download required metadata files if not already available
for filename, file_id in FILES.items():
    file_path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(file_path):
        print(f"Downloading {filename}...")
        gdown.download(f"https://drive.google.com/uc?id={file_id}", file_path, quiet=False)
        print(f"{filename} downloaded.")

print("All files are ready.")
