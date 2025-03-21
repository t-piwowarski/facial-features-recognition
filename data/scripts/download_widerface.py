import os
import gdown
import zipfile

# Path to the directory where WIDERFace data will be stored
DATA_DIR = "data/WIDERFace"
os.makedirs(DATA_DIR, exist_ok=True)

# Google Drive file IDs for WIDERFace dataset components
FILES = {
    "WIDER_train.zip": "1TGu9uoFDFMWvA_B4EvM60qHdXXscUAZJ",   # Training images
    "wider_face_split.zip": "1fZjLVbT5TA2ewU5ohsPfkgxfZ5K6XYNx",  # Annotations
}

# Download a file from Google Drive using gdown
def download_file(file_name, file_id):
    file_path = os.path.join(DATA_DIR, file_name)
    
    if not os.path.exists(file_path):
        print(f"Downloading {file_name}...")
        gdown.download(f"https://drive.google.com/uc?id={file_id}", file_path, quiet=False)
    else:
        print(f"{file_name} already exists. Skipping download.")

# Extract a ZIP archive to a specified directory and delete the archive afterwards
def extract_zip(zip_path, extract_to):
    print(f"Extracting: {zip_path} → {extract_to}")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
    os.remove(zip_path)  # Clean up zip file after extraction

# Download and extract each required file
for file_name, file_id in FILES.items():
    download_file(file_name, file_id)
    extract_zip(os.path.join(DATA_DIR, file_name), DATA_DIR)

print("WIDERFace dataset is ready.")
