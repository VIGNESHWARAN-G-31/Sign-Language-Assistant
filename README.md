# ✋🤖 Sign Language Detection using Faster R-CNN

This project uses a deep learning-based object detection model — **Faster R-CNN with ResNet-50 FPN** — to detect and classify hand signs representing digits (`1–9`) and letters (`A–Z`) in real-time from your webcam. 🎯📷

---

## 🚀 Features

- 🎥 Real-time webcam detection
- 🔠 Recognizes signs for numbers `1-9` and letters `A-Z`
- 🧠 Powered by PyTorch + Torchvision
- 🖥️ Runs on CPU (GPU optional for faster inference)

---

## 🧠 Model Details

- **Architecture**: Faster R-CNN (ResNet-50 FPN)
- **Framework**: PyTorch
- **Input**: RGB webcam frame
- **Output**: Bounding boxes + class labels
- **Classes**: 35 (Digits + Alphabets)

--
##🔤 Class Mapping

-🔢 Digits: 1 to 9

-🔡 Alphabets: A to Z

-🧾 Total: 35 classes (excluding background)

--
##📚 Dataset Used

-Dataset: Indian Sign Language Detection
-Source: Roboflow Universe
-Classes: Indian sign gestures for 1–9 and A–Z
-Format: COCO JSON annotations

#Preprocessing:

-Resized images to a standard resolution

-Applied augmentations (rotation, flipping, brightness, etc.)

-You can download the dataset from Roboflow, convert it to COCO format, and use it directly for training object detection models like Faster R-CNN.

