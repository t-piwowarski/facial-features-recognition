# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 00:34:20 2025

@author: tomas
"""

import torchvision.datasets as datasets
import os

# 📂 Główna ścieżka do danych
DATA_DIR = "data/WIDERFace"
os.makedirs(DATA_DIR, exist_ok=True)

# 📥 Pobieranie zbioru WIDERFace
def download_widerface():
    print(f"📥 Pobieranie WIDERFace")
    datasets.WIDERFace(root=DATA_DIR, split='train', download=True)
    print(f"✅ Pobieranie zakończone")
    
print("🎉 Pobieranie WIDERFace zakończone!")
