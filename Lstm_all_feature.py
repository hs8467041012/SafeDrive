import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import tensorflow as tf

# ----------------- LOAD TRAINED MODEL -----------------
model = tf.keras.models.load_model("lstm_all_feature_05_septmodel.h5")

# ----------------- PARAMETERS -----------------
SEQUENCE_LENGTH = 10   # must match training seq_len
EAR_THRESHOLD = 0.25
MOR_THRESHOLD = 0.045
NLR_THRESHOLD = 0.02

# ----------------- MEDIAPIPE SETUP -----------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Landmark indices (MediaPipe FaceMesh)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [61, 291, 81, 178, 13, 14]  # corners + top/bottom lips
NOSE = [1]  # tip of the nose for NLR reference
LEFT_IRIS = [468]   # iris center left
RIGHT_IRIS = [473]  # iris center right

# Store sequence of features [EAR, MOR, NLR]
feature_sequence = deque(maxlen=SEQUENCE_LENGTH)

# ----------------- FEATURE FUNCTIONS -----------------
def euclidean_dist(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def compute_ear(landmarks, eye_indices, w, h):
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_indices]
    A = euclidean_dist(pts[1], pts[5])
    B = euclidean_dist(pts[2], pts[4])
    C = euclidean_dist(pts[0], pts[3])
    ear = (A + B) / (2.0 * C)
    return ear

def compute_mor(landmarks, mouth_indices, w, h):
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in mouth_indices]
    A = euclidean_dist(pts[2], pts[3])   # top–bottom lip
    C = euclidean_dist(pts[0], pts[1])   # mouth corners
    mor = A / C if C != 0 else 0
    return mor

def compute_nlr(landmarks, nose_idx, iris_idx, w, h):
    nose = (int(landmarks[nose_idx].x * w), int(landmarks[nose_idx].y * h))
    iris = (int(landmarks[iris_idx].x * w), int(landmarks[iris_idx].y * h))
    dist = euclidean_dist(nose, iris)
    return dist / w  # normalize by width of frame

# ----------------- VIDEO CAPTURE -----------------
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Compute EAR
            left_ear = compute_ear(face_landmarks.landmark, LEFT_EYE, w, h)
            right_ear = compute_ear(face_landmarks.landmark, RIGHT_EYE, w, h)
            ear = (left_ear + right_ear) / 2.0

            # Compute MOR
            mor = compute_mor(face_landmarks.landmark, MOUTH, w, h)

            # Compute NLR (average of left and right iris distance to nose)
            nlr_left = compute_nlr(face_landmarks.landmark, NOSE[0], LEFT_IRIS[0], w, h)
            nlr_right = compute_nlr(face_landmarks.landmark, NOSE[0], RIGHT_IRIS[0], w, h)
            nlr = (nlr_left + nlr_right) / 2.0

            # Add features to sequence
            feature_sequence.append([ear, mor, nlr])

            # Prediction only if we have full sequence
            if len(feature_sequence) == SEQUENCE_LENGTH:
                X_test = np.array(feature_sequence).reshape(1, SEQUENCE_LENGTH, 3)
                y_pred = model.predict(X_test, verbose=0)[0][0]

                if y_pred > 0.5:  # threshold may be tuned
                    cv2.putText(frame, "DROWSY ALERT!", (50, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                else:
                    cv2.putText(frame, "Awake", (50, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

            # Draw landmarks (optional for debugging)
            mp_drawing.draw_landmarks(
                frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1)
            )

    cv2.imshow("Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
