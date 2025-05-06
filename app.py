import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import cv2
import numpy as np

# Define the model loader
def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# Define class label mapping: 0–8 => '1'–'9', 9–35 => 'A'–'Z'
class_names = [str(i) for i in range(1, 10)] + [chr(i) for i in range(ord('A'), ord('Z')+1)]

# Load model
num_classes = 36  # 9 digits + 26 letters + 1 background (implicit)
model = get_model(num_classes)
model.load_state_dict(torch.load(r"D:\Porjects\Sign Language\sign_language_detector.pth", map_location='cpu'))
model.eval()

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print(" Failed to open webcam.")
    exit()
print(" Webcam started. Press 'q' to quit.")

threshold = 0.7

while True:
    ret, frame = cap.read()
    if not ret:
        print(" Failed to grab frame.")
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_tensor = torchvision.transforms.functional.to_tensor(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image_tensor)

    boxes = outputs[0]['boxes']
    labels = outputs[0]['labels']
    scores = outputs[0]['scores']

    for box, label, score in zip(boxes, labels, scores):
        if score > threshold and label.item() < len(class_names):
            x1, y1, x2, y2 = box.int().tolist()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            class_name = class_names[label.item()]
            label_text = f"{class_name} ({score:.2f})"
            cv2.putText(frame, label_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

    cv2.imshow("Sign Language Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
