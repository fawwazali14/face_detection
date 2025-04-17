import cv2
from deepface import DeepFace
import numpy as np

# === Configuration ===
KNOWN_FACES_DIR = "known_faces"
MODEL_NAME = "ArcFace"
DISTANCE_METRIC = "euclidean"
THRESHOLD = 4.8
FRAME_SKIP = 5
MEMORY_DURATION = 30  # Frames to keep label if face disappears

# === Load Haar Cascade ===
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# === Webcam Init ===
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("❌ Cannot open webcam.")
    exit()

print("📷 Webcam started... press 'q' to quit.")
frame_count = 0

# === Store tracked identities: [((x, y, w, h), identity, last_seen_frame)] ===
tracked_faces = []

# === Main Loop ===
while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Frame capture failed.")
        continue

    frame_count += 1
    display_frame = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    updated_faces = []

    if frame_count % FRAME_SKIP == 0:
        for (x, y, w, h) in faces:
            identity = "Detecting..."
            face_img = frame[y:y+h, x:x+w]

            try:
                rgb_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                results = DeepFace.find(
                    img_path=rgb_face,
                    db_path=KNOWN_FACES_DIR,
                    model_name=MODEL_NAME,
                    distance_metric=DISTANCE_METRIC,
                    enforce_detection=False
                )

                if len(results[0]) > 0:
                    best = results[0].iloc[0]
                    distance = best['distance']
                    name_guess = best['identity'].split('/')[-1].split('.')[0]
                    if distance < THRESHOLD:
                        identity = name_guess
                        print(f"✅ Match: {identity} (distance: {distance:.2f})")
                    else:
                        identity = "Unknown"
                else:
                    identity = "Unknown"
            except Exception as e:
                identity = "Error"
                print(f"⚠️ DeepFace error: {e}")

            updated_faces.append(((x, y, w, h), identity, frame_count))

        tracked_faces = updated_faces
    else:
        # Use last known identities
        updated_faces = [face for face in tracked_faces if frame_count - face[2] <= MEMORY_DURATION]

    # Draw boxes and labels
    for (x, y, w, h), identity, _ in updated_faces:
        cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(display_frame, identity, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Show result
    cv2.imshow("🔍 DeepFace Multi-Face Recognition", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# === Cleanup ===
cap.release()
cv2.destroyAllWindows()
print("👋 Webcam closed.")
