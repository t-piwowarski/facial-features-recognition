# Facial Features Recognition

This project is used to recognize facial features, including gender and smile, using deep learning models (CNN and ResNet50). The project uses CelebA and WIDERFace datasets to train and test models. Additionally, they enable real-time facial feature detection. The code was written in Python using PyTorch and PyTorch Lightning.

---

## 🏗️ Model architecture

### Gender Recognition Model (**GenderCNN**)

- A convolutional neural network (CNN) consisting of four blocks:
  - Layers **Conv2D**, **ReLU**, **BatchNorm**, **MaxPooling**
  - Fully connected classification layers
- Normalizing data to a range **[-1,1]**
- Augmentation: reflections, rotation **(-15° do 15°)**, shift **(0-10% wymiaru)**, changes in brightness, contrast, saturation and hue
- **Early stopping** (stopping training after 5 epochs without improvement)

### A model that recognizes a smile (**SmileResNet**)

- **ResNet50** with transfer learning
- **Fine-tuning** 4 last layers
- Same **augmentation** process as in **GenderCNN**

---

## 📊 Model results

### Results on CelebA set:

#### Gender recognition:

- **Accuracy**: 98%
- **Precision**: 97% (men), 98% (women)
- **Sensitivity**: 96% (men), 98% (women)
- **F1-score**: 97% (men), 98% (women)

<img src="docs/images/celeba_gender_confusion_matrix.png" width="500">

<img src="docs/images/celeba_gender_detection_result.png" width="500">

#### Smile recognition:

- **Accuracy**: 91%
- **Precision**: 90% (non-smiling), 92% (smiling)
- **Sensitivity**: 93% (non-smiling), 90% (smiling)
- **F1-score**: 92% (non-smiling), 91% (smiling)

<img src="docs/images/celeba_smile_confusion_matrix.png" width="500">

<img src="docs/images/celeba_smile_detection_result.png" width="500">

### Results on the WIDERFace set:

<img src="docs/images/widerface_detection.jpg" width="500">

#### Gender recognition:

- **Accuracy**: 85%
- **Precision**: 88% (men), 82% (women)
- **Sensitivity**: 82% (men), 89% (women)
- **F1-score**: 85% (men), 85% (women)

<img src="docs/images/widerface_gender_confusion_matrix.png" width="500">

#### Smile recognition:

- **Accuracy**: 84%
- **Precision**: 76% (non-smiling), 92% (smiling)
- **Sensitivity**: 92% (non-smiling), 77% (smiling)
- **F1-score**: 83% (non-smiling), 84% (smiling)

<img src="docs/images/widerface_smile_confusion_matrix.png" width="500">

---

## 📂 Repository structure

facial-features-recognition\
│── src\
│  │── models\
│  │  │── gender_cnn.py\
│  │  │── resnet50_smile.py\
│  │── training\
│  │  │── train_gender.py\
│  │  │── train_smile.py\
│  │── inference\
│  │  │── test_widerface.py\
│  │  │── webcam_detection.py\
│ \
│── data\
│  │── scripts\
│  │  │── download_celeba.py\
│  │  │── download_widerface.py\
│  │── selected_with_bboxes.txt\
│ \
│── saved_models\
│  │── gender_cnn.pth\
│  │── resnet50_smile.pth\
│ \
├── docs\
│   │── images\
│   │   │── gender_cnn_architecture.png\
│   │   │── resnet_smile_architecture.png\
│   │   │── gender_confusion_matrix_celeba.png\
│   │   │── smile_confusion_matrix_celeba.png\
│   │   │── gender_confusion_matrix_widerface.png\
│   │   │── smile_confusion_matrix_widerface.png\
│ \
│── README.md\
│── .gitignore\
│── requirements.txt\

---

## 🚀 Installation

1. **Clone repository:**

   ```bash
   git clone https://github.com/TwojUserName/facial-features-recognition.git
   cd facial-features-recognition

2. **Create and activate a virtual environment (optional but recommended):**
   
- On Windows:
     
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```
   
- On Linux/macOS:
     
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install the required packages:**
   
   ```bash
   pip install -r requirements.txt
   ```

---

## 📥 Downloading data

### CelebA

   Use CelebA download script that downloads images and text files from Google Drive:
   
   ```bash
   python data/scripts/download_celeba.py
   ```

   *The files will be saved in `data/celeba/` folder.*
   
> **Note:** If you experience problems with automatic downloading via `gdown`, download the files manually (e.g. from Google Drive or another source) and place them in the appropriate folders.

### WIDERFace (only training data)

   Use the script to download the WIDERFace (training) dataset and annotations:
   
   ```bash
   python data/scripts/download_widerface.py
   ```

   *Files will be saved in `data/WIDERFace/` folder.*
   
> **Note:** If you experience problems with automatic downloading via 'gdown', please download the files manually (e.g. from Kaggle or another source) and place them in the appropriate folders.

---

## ⚙️ Running models

### Training

- **Gender Recognition Model Training:**

   ```bash
   python src/training/train_gender.py
   ```

- **Smile Detection Model Training:**

  ```bash
  python src/training/train_smile.py
  ```

### Testing

- **Testing on the WIDERFace set:**

  ```bash
  python src/inference/test_widerface.py
  ```

- **Real-time prediction (camera):**

  ```bash
  python src/inference/webcam_detection.py
  ```

---

## 📚 Zawartość repozytorium

- `src/models/`: Contains model definitions, including `GenderCNNLightning` and `SmileResNetLightning`.
- `src/training/`: Training scripts for models (gender and smile).
- `src/inference/`: Scripts for testing models - both on the WIDERFace dataset and in real time.
- `data/scripts/`: Scripts for automatic downloading and unpacking of data (CelebA and WIDERFace).
- `data/selected_with_bboxes.txt`: Annotation file, used for testing models on the WIDERFace dataset.
- `saved_models/`: Folder for storing saved models.
- `requirements.txt`: List of dependencies necessary to run the project.
- `.gitignore`: Configuration to exclude unnecessary files (e.g. data, virtual environments, cache files).

