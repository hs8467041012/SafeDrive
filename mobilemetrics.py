import numpy as np
import cv2
import tensorflow as tf
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ----------------- LOAD MODEL -----------------
model = tf.keras.models.load_model("mobilenet_drowsy_detector.h5", compile=False)
IMG_SIZE = (224, 224)

# ----------------- GROUND TRUTH PLACEHOLDER -----------------
# 👉 Since webcam has no ground-truth labels, you need to annotate manually
# Example: Press "a" for Alert, "d" for Drowsy while testing yourself.
y_true, y_pred = [], []

cap = cv2.VideoCapture(0)  # webcam

print("Press 'a' for Alert ground truth, 'd' for Drowsy ground truth, 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Preprocess frame
    img = cv2.resize(frame, IMG_SIZE)
    img = np.expand_dims(img, axis=0) / 255.0
    
    # Prediction
    pred = model.predict(img, verbose=0)[0][0]
    label = "Drowsy" if pred > 0.5 else "Alert"
    color = (0,0,255) if label=="Drowsy" else (0,255,0)

    # Display label
    cv2.putText(frame, label, (30,50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow("SafeDrive - Drowsiness Detection", frame)

    # Key press handling
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):  # quit
        break
    elif key == ord("a"):  # ground truth: Alert
        y_true.append(0)  
        y_pred.append(0 if label=="Alert" else 1)
    elif key == ord("d"):  # ground truth: Drowsy
        y_true.append(1)
        y_pred.append(1 if label=="Drowsy" else 0)

cap.release()
cv2.destroyAllWindows()

# ----------------- EVALUATION -----------------
if y_true and y_pred:
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print("\n📊 Live Video Evaluation Results:")
    print(f"Accuracy: {acc:.2f}")
    print(f"Precision: {prec:.2f}")
    print(f"Recall: {rec:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print("Confusion Matrix:\n", cm)

    # ----------------- SAVE TO CSV -----------------
    metrics = {
        "Accuracy": [acc],
        "Precision": [prec],
        "Recall": [rec],
        "F1 Score": [f1],
        "True_Positive": [cm[1,1]],
        "True_Negative": [cm[0,0]],
        "False_Positive": [cm[0,1]],
        "False_Negative": [cm[1,0]]
    }
    df_metrics = pd.DataFrame(metrics)
    df_metrics.to_csv("mobile_live_video_metrics.csv", index=False)
    print("\n✅ Metrics saved to mobile_live_video_metrics.csv")

else:
    print("⚠️ No predictions were recorded. Try recording longer with ground truth annotations.")
