import os
import gdown
import zipfile

# 📂 Ścieżka do folderu z danymi
DATA_DIR = "data/WIDERFace"
os.makedirs(DATA_DIR, exist_ok=True)

# 🔗 Google Drive IDs dla WIDERFace
FILES = {
    "WIDER_train.zip": "1TGu9uoFDFMWvA_B4EvM60qHdXXscUAZJ",  # Pliki treningowe
    "wider_face_split.zip": "1fZjLVbT5TA2ewU5ohsPfkgxfZ5K6XYNx",  # Anotacje
}

# 📥 Pobieranie plików przez gdown
def download_file(file_name, file_id):
    file_path = os.path.join(DATA_DIR, file_name)
    
    if not os.path.exists(file_path):
        print(f"📥 Pobieranie {file_name}...")
        gdown.download(f"https://drive.google.com/uc?id={file_id}", file_path, quiet=False)
    else:
        print(f"✅ {file_name} już istnieje. Pomijam pobieranie.")

# 📦 Rozpakowywanie ZIP
def extract_zip(zip_path, extract_to):
    print(f"📂 Rozpakowywanie: {zip_path} → {extract_to}")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
    os.remove(zip_path)  # 🗑️ Usuwamy ZIP po rozpakowaniu

# 🚀 Pobieranie i rozpakowywanie plików
for file_name, file_id in FILES.items():
    download_file(file_name, file_id)
    extract_zip(os.path.join(DATA_DIR, file_name), DATA_DIR)

print("🎉 Pobieranie i rozpakowywanie WIDERFace zakończone!")
