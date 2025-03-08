# Facial Features Recognition

Projekt ten służy do rozpoznawania cech twarzy, w tym płci oraz uśmiechu, przy użyciu modeli głębokiego uczenia (CNN oraz ResNet50). W projekcie wykorzystujemy zbiory CelebA oraz WIDERFace do trenowania i testowania modeli. Kod został napisany w Pythonie przy użyciu PyTorch i PyTorch Lightning.

---

## 📂 Struktura Repozytorium

facial-features-recognition\
├── src\
│ ├── models\
│ │ ├── gender_cnn.py # Definicja modelu CNN do rozpoznawania płci\
│ │ ├── resnet50_smile.py # Definicja modelu ResNet50 do wykrywania uśmiechu\
│ ├── training\
│ │ ├── train_gender.py # Skrypt do treningu modelu płci\
│ │ ├── train_smile.py # Skrypt do treningu modelu uśmiechu\
│ ├── inference\
│ │ ├── test_widerface.py # Testowanie modeli na zbiorze WIDERFace\
│ │ ├── webcam_detection.py # Predykcja cech twarzy w czasie rzeczywistym przy użyciu kamery\
│\
├── data\
│ ├── scripts\
│ │ ├── download_celeba.py # Pobieranie i rozpakowywanie zbioru CelebA (obrazy i metadane)\
│ │ ├── download_widerface.py # Pobieranie i rozpakowywanie zbioru WIDERFace (trening) wraz z anotacjami\
│ ├── selected_with_bboxes.txt # Anotacje wybranych obrazów ze zbioru WIDERFace\
│\
├── saved_models # Folder na zapisane modele (np. .pth)\
│ ├── gender_cnn.pth\
│ ├── resnet50_smile.pth\
│\
├── .gitignore\
├── requirements.txt\
├── README.md\

---

## 🚀 Instalacja

1. **Sklonuj repozytorium:**

   ```bash
   git clone https://github.com/TwojUserName/facial-features-recognition.git
   cd facial-features-recognition

2. **Utwórz i aktywuj środowisko wirtualne (opcjonalnie, ale zalecane):**
   - Na Windows:

