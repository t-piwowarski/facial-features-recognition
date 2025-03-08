# Facial Features Recognition

Projekt ten służy do rozpoznawania cech twarzy, w tym płci oraz uśmiechu, przy użyciu modeli głębokiego uczenia (CNN oraz ResNet50). W projekcie wykorzystujemy zbiory CelebA oraz WIDERFace do trenowania i testowania modeli. Kod został napisany w Pythonie przy użyciu PyTorch i PyTorch Lightning.

---

## 📂 Struktura Repozytorium

facial-features-recognition/ 
├── src/
│   ├── models/
│   │   ├── gender_cnn.py         # Model do rozpoznawania płci
│   │   ├── resnet50_smile.py     # Model do wykrywania uśmiechu
│   ├── training/
│   │   ├── train_gender.py       # Skrypt do trenowania modelu płci
│   │   ├── train_smile.py        # Skrypt do trenowania modelu uśmiechu
│   ├── inference/
│   │   ├── test_widerface.py     # Skrypt do testowania na zbiorze WIDERFace
│   │   ├── webcam_detection.py   # Wykrywanie na kamerze w czasie rzeczywistym
│
├── data/
│   ├── scripts/
│   │   ├── download_celeba.py    # Skrypt do pobierania zbioru CelebA
│   │   ├── download_widerface.py # Skrypt do pobierania zbioru WIDERFace
│   ├── selected_with_bboxes.txt  # Anotacje do zbioru WIDERFace
│
├── saved_models/                 # (opcjonalnie) Folder na zapisane modele
│   ├── gender_cnn.pth
│   ├── resnet50_smile.pth
│
├── README.md
├── .gitignore
├── requirements.txt

---

## 🚀 Instalacja

1. **Sklonuj repozytorium:**

   ```bash
   git clone https://github.com/TwojUserName/facial-features-recognition.git
   cd facial-features-recognition

2. **Utwórz i aktywuj środowisko wirtualne (opcjonalnie, ale zalecane):**
   - Na Windows:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```
   - Na Linux/macOS:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Zainstaluj wymagane pakiety:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 📥 Pobieranie Danych

## CelebA
   Użyj skryptu do pobierania CelebA, który pobiera obrazy i pliki tekstowe z Google Drive:
   ```bash
   python data/scripts/download_celeba.py
   ```
   *Pliki zostaną zapisane w folderze <mark>data/celeba/</mark>.*\
\
## WIDERFace (tylko dane treningowe)
   Użyj skryptu do pobierania zbioru WIDERFace (trening) oraz anotacji:
   ```bash
   python data/scripts/download_widerface.py
   ```
   *Pliki zostaną zapisane w folderze <mark>data/WIDERFace/</mark>.*\
> **Uwaga:** W przypadku problemów z automatycznym pobieraniem przez gdown, pobierz pliki ręcznie (np. z Kaggle lub innego źródła) i umieść je w odpowiednich folderach.


