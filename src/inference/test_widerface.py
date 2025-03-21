import os
import cv2
import torch
from torchvision import transforms
from PIL import Image
from src.models.gender_cnn import GenderCNNLightning
from src.models.resnet50_smile import SmileResNetLightning


# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pretrained gender classification model
gender_model = GenderCNNLightning().to(device)
gender_model.load_state_dict(torch.load("saved_models/gender_cnn"))
gender_model.eval()

# Load pretrained smile detection model
smile_model = SmileResNetLightning().to(device)
smile_model.load_state_dict(torch.load("saved_models/smile_resnet.pth"))
smile_model.eval()

# Image preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((178, 178)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Paths to data and annotations
data_dir = "data/WIDER_train/images"
annotation_file = "data/selected_with_bboxes.txt"

# Parse annotation file into structured format
def parse_widerface_annotations(annotation_path):
    with open(annotation_path, "r") as f:
        lines = f.readlines()

    data = []
    i = 0
    while i < len(lines):
        image_path = lines[i].strip()
        num_faces = int(lines[i + 1].strip())
        face_data = []

        for j in range(num_faces):
            x, y, w, h, gender, smile = lines[i + 2 + j].strip().split(" ")
            face_data.append({
                "bbox": [int(x), int(y), int(w), int(h)],
                "gender": gender,
                "smile": smile
            })
        
        data.append({"image_path": image_path, "faces": face_data})
        i += 2 + num_faces
    
    return data

# Load annotations
annotations = parse_widerface_annotations(annotation_file)

# Create output directory for annotated images
output_dir = "output_predictions"
os.makedirs(output_dir, exist_ok=True)

# Initialize evaluation counters
correct_gender = 0
correct_smile = 0
total_faces = 0

gender_true = []
gender_predicted = []
smile_true = []
smile_predicted = []

# Process each image and run predictions
for entry in annotations:
    image_path = os.path.join(data_dir, entry["image_path"])
    image = cv2.imread(image_path)

    if image is None:
        print(f"Nie znaleziono obrazu: {image_path}")
        continue

    for face in entry["faces"]:
        x, y, w, h = face["bbox"]

        # Crop and preprocess face
        cropped_face = image[y:y+h, x:x+w]
        cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(cropped_face)
        tensor_image = transform(pil_image).unsqueeze(0).to(device)

        # Run predictions
        gender_output = gender_model(tensor_image)
        smile_output = smile_model(tensor_image)

        gender_pred = torch.argmax(gender_output, dim=1).item()
        smile_pred = torch.argmax(smile_output, dim=1).item()

        gender_label = "Male" if gender_pred == 1 else "Female"
        smile_label = "Smiling" if smile_pred == 1 else "No_Smiling"

        # Track accuracy
        gender_correct = (gender_label == face["gender"])
        smile_correct = (smile_label == face["smile"])

        gender_true.append(face["gender"] == "Male")
        gender_predicted.append(gender_pred == 1)
        smile_true.append(face["smile"] == "Smiling")
        smile_predicted.append(smile_pred == 1)

        correct_gender += gender_correct
        correct_smile += smile_correct
        total_faces += 1

        # Draw predictions on image
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0) if gender_correct and smile_correct else (0, 0, 255), 2)
        cv2.putText(image, gender_label, (x, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(image, smile_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Save annotated image
    output_path = os.path.join(output_dir, os.path.basename(entry["image_path"]))
    cv2.imwrite(output_path, image)

# Final accuracy stats
accuracy_gender = correct_gender / total_faces if total_faces > 0 else 0
accuracy_smile = correct_smile / total_faces if total_faces > 0 else 0

print(f"Gender classification accuracy: {accuracy_gender:.2f}")
print(f"Smile detection accuracy: {accuracy_smile:.2f}")
