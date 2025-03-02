import os
import requests
import zipfile

DATASET_URL = "https://s3.amazonaws.com/content.udacity-data.com/courses/nd089/CelebA_128.zip"
DATA_DIR = "data/raw"

def download_and_extract(url, dest_folder):
    os.makedirs(dest_folder, exist_ok=True)
    zip_path = os.path.join(dest_folder, "celeba.zip")

    print(f"📥 Pobieranie zbioru danych z {url}...")
    response = requests.get(url, stream=True)
    with open(zip_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
    
    print("📦 Rozpakowywanie danych...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(dest_folder)
    
    os.remove(zip_path)
    print("✅ Pobieranie zakończone!")

if __name__ == "__main__":
    download_and_extract(DATASET_URL, DATA_DIR)
