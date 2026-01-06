import cv2
import numpy as np
from collections import deque
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------- CONFIG -----------------
SEQUENCE_LENGTH = 10
IMG_SIZE = 64
model_path = "cnn_lstm_all_features_06_OCT_drowsiness.h5"

# ----------------- LOAD MODEL -----------------
model = load_model(model_path)
print("✅ Model Loaded Successfully!")
print("Expected Input Shape:", model.input_shape)

# ----------------- FRAME PREPROCESSING -----------------
def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    normalized = resized.astype("float32") / 255.0
    return normalized.reshape(IMG_SIZE, IMG_SIZE, 1)

# ----------------- VIDEO STREAM -----------------
cap = cv2.VideoCapture(0)
frame_buffer = deque(maxlen=SEQUENCE_LENGTH)
aux_buffer = deque(maxlen=SEQUENCE_LENGTH)

y_true, y_pred, pred_probs = [], [], []

print("\nControls:")
print("Press 'a' → AWAKE (ground truth)")
print("Press 'd' → DROWSY (ground truth)")
print("Press 'q' → quit and show metrics\n")

last_truth = "None"

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Frame not captured. Exiting.")
        break

    processed = preprocess_frame(frame)
    frame_buffer.append(processed)
    aux_buffer.append([0.0, 0.0, 0.0])  # Dummy features

    label = "Waiting..."
    pred = 0.0

    if len(frame_buffer) == SEQUENCE_LENGTH:
        input_sequence = np.expand_dims(np.array(frame_buffer), axis=0)
        aux_input = np.expand_dims(np.array(aux_buffer), axis=0).astype(np.float32)

        try:
            pred = model.predict([input_sequence, aux_input], verbose=0)[0][0]
            label = "Drowsy" if pred > 0.5 else "Awake"
        except Exception as e:
            print("Prediction error:", e)
            continue

    # Display predictions
    cv2.putText(frame, f"Model: {label} ({pred:.2f})", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 0, 255) if label == "Drowsy" else (0, 255, 0), 2)

    cv2.putText(frame, f"Truth: {last_truth}", (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    cv2.imshow("CNN+LSTM Drowsiness Detection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('a'):
        y_true.append(0)
        y_pred.append(0 if pred <= 0.5 else 1)
        pred_probs.append(pred)
        last_truth = "Awake"
    elif key == ord('d'):
        y_true.append(1)
        y_pred.append(0 if pred <= 0.5 else 1)
        pred_probs.append(pred)
        last_truth = "Drowsy"
    elif key == ord('q'):
        print("Exiting...")
        break

cap.release()
cv2.destroyAllWindows()

# ----------------- METRICS -----------------
# ----------------- ADVANCED PLOT (LIKE SCREENSHOT) -----------------
if len(y_true) > 0:
    print("\n📊 Evaluation Metrics:")
    acc = accuracy_score(y_true, y_pred)
    print("Accuracy:", acc)
    print("\nClassification Report:\n", classification_report(y_true, y_pred, target_names=["Awake", "Drowsy"]))
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:\n", cm)

    # Save CSV
    df = pd.DataFrame({
        "Frame": np.arange(len(y_true)),
        "True": y_true,
        "Pred": y_pred,
        "Confidence": pred_probs
    })
    df.to_csv("drowsiness_results.csv", index=False)
    print("\n✅ Saved predictions to drowsiness_results.csv")

    # ----------------- NEW GRAPH STYLE (LIKE YOUR IMAGE) -----------------
    plt.figure(figsize=(12, 6))
    plt.title("Confidence Scores Over Time")
    plt.xlabel("Frame")
    plt.ylabel("Confidence")

    # Plot Awake (Pred=0)
    awake_indices = df[df["Pred"] == 0].index
    plt.plot(awake_indices, df.loc[awake_indices, "Confidence"], 
             'o--', color='royalblue', label='Awake', linewidth=1.5, markersize=5)

    # Plot Drowsy (Pred=1)
    drowsy_indices = df[df["Pred"] == 1].index
    plt.plot(drowsy_indices, df.loc[drowsy_indices, "Confidence"], 
             'o--', color='darkorange', label='Drowsy', linewidth=1.5, markersize=5)

    plt.ylim(0.45, 1.05)
    plt.legend()
    plt.grid(False)
    plt.tight_layout()

    plt.savefig("confidence_over_time.png", dpi=300)
    plt.show()

    print("\n🖼️ Graph saved as 'confidence_over_time.png'.")
else:
    print("No ground truth labels collected. Metrics cannot be computed.")
