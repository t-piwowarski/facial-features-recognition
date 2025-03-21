import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # Fix for some environments using Intel MKL

import cv2
cv2.setNumThreads(0)  # Avoid OpenCV using multiple threads (can cause lag with live video)

import torch
from torchvision import transforms
from skimage.feature import Cascade
from src.models.gender_cnn import GenderCNNLightning
from src.models.resnet50_smile import SmileDetectionModel

# Select device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pretrained gender model
gender_model = GenderCNNLightning().to(device)
gender_model.load_state_dict(torch.load("saved_models/genderCNN.pth"))
gender_model.eval()

# Load pretrained smile model
smile_model = SmileDetectionModel().to(device)
smile_model.load_state_dict(torch.load("saved_models/smile_resnet.pth"))
smile_model.eval()

# Image preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((178, 178)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Face detection using skimage Cascade
def detect(frame, detector):
    detections = detector.detect_multi_scale(img=frame, scale_factor=1.2, step_ratio=1,
                                             min_size=(100, 100), max_size=(200, 200))
    boxes = []
    for detection in detections:
        x = detection['c']
        y = detection['r']
        w = detection['width']
        h = detection['height']
        
        # Adjust aspect ratio to match model input (178:218)
        desired_ratio = 178 / 218
        new_w = w
        new_h = int(w / desired_ratio)
        
        # Center the adjusted bounding box
        x = x - (new_w - w) // 2
        y = y - (new_h - h) // 2
        
        # Expand the box by 20% in all directions
        x = int(x - 0.5 * w)
        y = int(y - 0.5 * h)
        w = int(2 * w)
        h = int(2 * h)
        
        boxes.append((x, y, w, h))
    return boxes

# Run both models on a cropped face and return predictions
def predict_attributes(face):
    face = transform(face).unsqueeze(0)
    
    with torch.no_grad():
        gender_pred = gender_model(face)
        smile_pred = smile_model(face)
    
    gender = 'Male' if torch.argmax(gender_pred) == 1 else 'Female'
    smile = 'Smiling' if torch.argmax(smile_pred) == 1 else 'No smiling'
    
    return gender, smile

# Draw bounding boxes and predicted labels on the frame
def draw(frame, boxes):
    for x, y, w, h in boxes:
        face = frame[y:y+h, x:x+w]
        if face.size != 0:
            gender, smile = predict_attributes(face)
            label = f"{gender}, {smile}"
            cv2.rectangle(frame, (x, y), (x + w, y + h), color=(255, 0, 0), thickness=2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

# Main loop — live webcam classification
if __name__ == '__main__':
    file = "./face.xml"  # Haar-like cascade XML file (for skimage)
    detector = Cascade(file)
    cap = cv2.VideoCapture(0)
    skip = 5   # How often to re-run detection (to save performance)
    i = 0
    boxes = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect faces every 'skip' frames
        if i % skip == 0:
            boxes = detect(frame, detector)
        draw(frame, boxes)
        cv2.imshow('Detekcja Twarzy, Płci i Uśmiechu', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        i += 1
    
    cap.release()
    cv2.destroyAllWindows()
