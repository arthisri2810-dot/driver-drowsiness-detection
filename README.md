# driver-drowsiness-detection
Driver Drowsiness Detection using Deep Learning
🚀 Overview

Driver fatigue is a major cause of road accidents due to reduced alertness and slow reaction time.
This project presents a non-intrusive, vision-based system that detects driver drowsiness using eye closure and yawning analysis with deep learning.

🎯 Problem Statement

Traditional drowsiness detection systems rely on:

Wearable sensors
Vehicle behavior

These methods are often intrusive and unreliable.
This project solves the problem using facial image analysis to detect:

👁️ Eye Closure
😮 Yawning
💡 Solution Approach

We built a Computer Vision + Deep Learning system that:

Detects eye state (Open / Closed)
Detects mouth state (Yawn / No Yawn)
Combines both to classify fatigue level:
🟢 Alert
🟡 Mild Fatigue
🔴 Severe Fatigue
🛠️ Tech Stack
Python
TensorFlow / Keras
CNN (Convolutional Neural Networks)
MobileNetV2 (Transfer Learning)
OpenCV
NumPy
📂 Dataset

📌 Dataset Link:
👉 https://drive.google.com/file/d/1Rj0EYgDy0sy7jc7-KXvQiHwpqA0Dzj__/view?usp=sharing

Dataset Details
Classes:
Eyes Open
Eyes Closed
Yawn
No Yawn
Preprocessed:
Resized (224×224)
Normalized
Augmented (rotation, zoom, brightness)
⚙️ Project Workflow
1️⃣ Data Preparation
Image resizing & normalization
Data augmentation
Train / Validation / Test split
2️⃣ Model Development
Eye detection model
Yawn detection model
Transfer Learning using MobileNetV2
3️⃣ Model Training
Optimizer: Adam
Loss Function: Categorical Crossentropy
Metrics: Accuracy
4️⃣ Evaluation
Confusion Matrix
Accuracy, Precision, Recall
Test dataset validation
🧠 Fatigue Detection Logic
Model Output	Fatigue Level
Open + No Yawn	Alert
Yawn	Mild Fatigue
Closed	Severe Fatigue
📈 Fatigue Progression Analysis

The system simulates continuous driving by:

Processing sequential frames
Converting predictions into fatigue levels
Plotting fatigue over time

This helps identify:

Transition from Alert → Fatigue → Severe Fatigue
📊 Results
High accuracy in detecting eye and mouth states
Robust performance under different conditions
Explainable rule-based fatigue classification
📉 Performance Analysis
Works well under normal lighting conditions
Limitations:
Poor lighting
Occlusions (glasses, masks)
Extreme head angles
💼 Business Use Cases
🚗 Accident prevention systems
🚚 Fleet management monitoring
🤖 ADAS (Advanced Driver Assistance Systems)
🛡️ Insurance risk assessment
🚘 Smart & autonomous vehicles
📦 Project Deliverables
✔️ Google Colab Notebook
✔️ Trained Models
✔️ Dataset Structure
✔️ Evaluation Metrics
✔️ Documentation
