SafeDrive – Real-Time Driver Drowsiness Detection System

SafeDrive is an AI-powered real-time driver drowsiness detection system designed to enhance road safety by continuously monitoring a driver’s facial cues and alerting them before fatigue leads to accidents.

This repository contains all training notebooks, deep learning models, feature engineering scripts, and real-time inference code used during development and experimentation.

Project Overview
Driver drowsiness is one of the leading causes of road accidents worldwide. SafeDrive leverages Computer Vision and Deep Learning techniques to detect early signs of fatigue such as:

Eye closure
Yawning
Nose Bending

Facial landmark-based fatigue metrics

The system is capable of real-time inference using webcam or video input and triggers alerts when drowsiness is detected.

 Key Features
 Real-time driver monitoring

 YOLOv8-based face & eye detection
 Deep Learning models (CNN, LSTM, CNN+LSTM)
 Facial landmark-based manual features
 Multiple model experiments and comparisons
 Trained models included for direct testing
 Modular Python scripts for inference and metrics

🗂️ Repository Structure
bash
Copy code
SafeDrive/
│
├── Notebooks (Training & Experiments)
│   ├── Copy_of_CNN+LSTM_All_Feature.ipynb
│   ├── Copy_of_LSTM_All_Feature.ipynb
│   ├── Copy_of_MobileNetV2_All_Feature.ipynb
│   ├── Copy_of_ResNet50.ipynb
│   ├── Copy_of_Yolov8.ipynb
│   ├── Copy_of_Drowsiness_Detection_3_April_2025.ipynb
│   └── Manual_Feature_2025.ipynb
│
├── Trained Models
│   ├── best.pt                         # YOLOv8 trained weights
│   ├── lstm_all_feature_05_septmodel.h5
│   └── mobilenet_drowsy_detector.h5
│
├── Python Scripts (Real-Time & Metrics)
│   ├── yolov8.py                       # YOLOv8 real-time detection
│   ├── Lstm_all_feature.py             # LSTM-based prediction
│   ├── lstm_cnn_metric.py              # CNN+LSTM evaluation
│   ├── mobilemetrics.py                # MobileNet evaluation
│
└── README.md


🛠️ Technologies Used:
Python
OpenCV
YOLOv8
TensorFlow / Keras
CNN, LSTM, CNN+LSTM
MobileNetV2
ResNet50
Facial Landmark Analysi
NumPy, Pandas, Matplotlib

Models Implemented:

YOLOv8	Face & eye detection
CNN	Spatial feature extraction
LSTM	Temporal fatigue pattern learning
CNN + LSTM	Combined spatial-temporal modeling
MobileNetV2	Lightweight real-time inference
ResNet50	Deep feature extraction
Manual Features	EAR, MAR & facial ratios

 How to Run (Real-Time Testing)
1️⃣ Clone the Repository
bash
Copy code
git clone https://github.com/hs8467041012/SafeDrive.git
cd SafeDrive
2️⃣ Install Dependencies
bash
Copy code
pip install opencv-python tensorflow ultralytics numpy pandas
3️⃣ Run YOLOv8 Real-Time Detection
bash
Copy code
python yolov8.py
4️⃣ Run LSTM / CNN+LSTM Prediction
bash
Copy code
python Lstm_all_feature.py
(Ensure trained .h5 and .pt model files are present in the directory)

 Results & Performance
High accuracy in detecting prolonged eye closure and yawning

CNN+LSTM models outperform single-frame CNN models

MobileNetV2 provides efficient performance for real-time deployment

YOLOv8 ensures fast and accurate facial region detection

 Use Cases:


 Driver safety systems
 Fleet & commercial vehicle monitoring
 Smart vehicles & ADAS systems
 Research in fatigue and behavior analysis

Academic Relevance
This project is suitable for:

Final Year B.Tech / M.Tech Projects
AI / ML / Computer Vision Research
Real-Time Deep Learning Systems

Author
Himanshu Kumar Sukralia
B.Tech – Computer Science & Engineering
Major Project: SafeDrive – Real-Time Drowsiness Detection

Support
If you find this project helpful:

⭐ Star the repository

🍴 Fork it

 Use it for learning & research








