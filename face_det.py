import cv2
import numpy as np
from deepface import DeepFace
import speech_recognition as sr
import pyttsx3
import requests
import json
import threading

# === Configuration ===
KNOWN_FACES_DIR = "known_faces"
MODEL_NAME = "ArcFace"
DISTANCE_METRIC = "euclidean"
THRESHOLD = 4.8
FRAME_SKIP = 5
MEMORY_DURATION = 30
openai_api_key = "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"

# === OpenAI Setup ===
url = "https://api.openai.com/v1/chat/completions"
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {openai_api_key}"
}

# === Voice Setup ===
engine = pyttsx3.init()
recognizer = sr.Recognizer()
recognizer.energy_threshold = 400
recognizer.dynamic_energy_threshold = True
recognizer.pause_threshold = 1.7

# === Globals ===
stop_assistant = False
recognized_people = set()
greeted_people = set()
voice_ready = threading.Event()

# === Voice Assistant Function ===
def voice_loop():
    global stop_assistant
    while not stop_assistant:
        with sr.Microphone() as source:
            print("🎙️ Please speak (say 'stop' to end):")
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            try:
                audio = recognizer.listen(source)
                user_question = recognizer.recognize_google(audio)
                print("You said:", user_question)

                if user_question.lower() == 'stop':
                    print("🎤 Ending conversation.")
                    engine.say("Goodbye!")
                    engine.runAndWait()
                    stop_assistant = True
                    break
                elif "name them" in user_question.lower():
                    if recognized_people:
                        names = ", ".join(recognized_people)
                        response = f"I see: {names}"
                    else:
                        response = "I see no one I recognize."
                else:
                    data = {
                        "model": "gpt-3.5-turbo",
                        "messages": [
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": user_question}
                        ]
                    }
                    response_api = requests.post(url, headers=headers, json=data)
                    if response_api.status_code == 200:
                        response = response_api.json()['choices'][0]['message']['content']
                    else:
                        response = "I'm sorry, something went wrong with the response."
                print("Assistant:", response)
                engine.say(response)
                engine.runAndWait()

            except sr.UnknownValueError:
                print("Sorry, I could not understand the audio.")
            except sr.WaitTimeoutError:
                print("Listening timed out while waiting for phrase to start.")

# === Start Voice Thread ===
voice_thread = threading.Thread(target=voice_loop)
voice_thread.daemon = True
voice_thread.start()

# === Face Detection Setup ===
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("❌ Cannot open webcam.")
    exit()

print("📷 Webcam started... press 'q' to quit.")
frame_count = 0
tracked_faces = []

# === Main Loop ===
while not stop_assistant:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Frame capture failed.")
        continue

    frame_count += 1
    display_frame = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    updated_faces = []
    current_frame_people = set()

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
                        current_frame_people.add(identity)
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
        updated_faces = [face for face in tracked_faces if frame_count - face[2] <= MEMORY_DURATION]

    # Greeting logic
    new_faces = current_frame_people - greeted_people
    if new_faces:
        if len(new_faces) > 1:
            greet_msg = "Hey everyone, how are you all doing today?"
        else:
            greet_msg = f"Hello {list(new_faces)[0]}, Welcome"
        engine.say(greet_msg)
        engine.runAndWait()
        greeted_people.update(new_faces)
        recognized_people.update(current_frame_people)
        voice_ready.set()

    # Draw boxes and labels
    for (x, y, w, h), identity, _ in updated_faces:
        cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(display_frame, identity, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Show result
    cv2.imshow("🔍 DeepFace Multi-Face Recognition", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        stop_assistant = True
        break

# === Cleanup ===
cap.release()
cv2.destroyAllWindows()
print("👋 Webcam closed.")