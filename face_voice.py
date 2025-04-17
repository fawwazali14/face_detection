# --- Imports and Logging Setup ---
import cv2
from deepface import DeepFace
import numpy as np
import os
import threading
import speech_recognition as sr
import pyttsx3
import openai
import sys

# Redirect all stdout/stderr to a log file for comprehensive logging
log_file = open("log.txt", "w")
sys.stdout = log_file
sys.stderr = log_file

print("Initializing the DeepFace GPT-voice assistant...")

# --- Load Face Recognition Model and Database ---
# Use DeepFace with ArcFace model for face recognition
print("Loading DeepFace ArcFace model...")
model = DeepFace.build_model("ArcFace")  # load ArcFace model

# Path to the database of known faces (structured by person name folders)
db_path = "known_faces"
# Prepare embeddings for each known person in the database
known_embeddings = {}  # name -> embedding vector
if os.path.isdir(db_path):
    for person_name in os.listdir(db_path):
        person_dir = os.path.join(db_path, person_name)
        if os.path.isdir(person_dir):
            # Use the first image found in the person's folder to represent them
            img_files = [f for f in os.listdir(person_dir) if f.lower().endswith(('.jpg','.png','.jpeg'))]
            if img_files:
                img_path = os.path.join(person_dir, img_files[0])
                try:
                    # Get embedding vector for this face using ArcFace model
                    rep = DeepFace.represent(img_path=img_path, model_name="ArcFace", model=model, 
                                              detector_backend="opencv", enforce_detection=False)
                    if rep and len(rep) > 0:
                        known_embeddings[person_name] = rep[0]["embedding"]
                        print(f"Loaded face embedding for {person_name}")
                except Exception as e:
                    print(f"Could not load {person_name}'s face: {e}")
else:
    print(f"Database path '{db_path}' not found. No known faces loaded.")

# Data structures for recognized and greeted people
recognized_names = []        # list of names currently recognized in frame
greeted_names = set()        # set of names already greeted in this session

# Thread coordination events
stop_event = threading.Event()      # signals both threads to stop
greeting_event = threading.Event()  # indicates that a greeting has been done (voice can start)

# --- Voice Assistant Thread Function ---
def voice_assistant():
    """Thread function to handle voice interaction using speech recognition and GPT-3.5."""
    print("Voice thread: starting and waiting for initial greeting...")
    # Initialize speech recognizer and microphone
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    # Initialize text-to-speech engine
    tts_engine = pyttsx3.init()
    # Optionally, adjust TTS voice properties (rate, volume, etc.)
    
    # Wait until at least one greeting has occurred before starting interaction
    greeting_event.wait()  
    print("Voice thread: Greeting done, beginning to listen for user input.")
    # Calibrate ambient noise level for the microphone
    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
    print("Voice thread: Microphone calibrated to ambient noise.")

    # Prepare conversation context list for GPT (start with a system prompt if desired)
    conversation = [ {"role": "system", "content": "You are a helpful voice assistant."} ]
    
    # Start listening loop
    with mic as source:
        while not stop_event.is_set():
            audio_data = None
            try:
                # Listen for a phrase (with timeout to periodically check stop_event)
                audio_data = recognizer.listen(source, timeout=1)
            except sr.WaitTimeoutError:
                # No speech detected in the last second, loop and check again
                if stop_event.is_set():
                    break
                continue
            if audio_data is None:
                continue
            # Recognize speech using Google Speech Recognition (online)
            user_text = ""
            try:
                user_text = recognizer.recognize_google(audio_data)
                user_text = user_text.strip()
                print(f"User said: {user_text}")
            except Exception as e:
                print(f"Speech recognition error: {e}")
                continue
            if user_text == "":
                continue

            # Handle voice commands
            user_text_lower = user_text.lower()
            if "name them" in user_text_lower:
                # List currently recognized people
                if recognized_names:
                    names_list = ", ".join(recognized_names)
                    response_text = f"The people I see are: {names_list}."
                else:
                    response_text = "I don't see any known people right now."
                print(f"Assistant (listing names): {response_text}")
            elif user_text_lower in ["stop", "exit", "quit"]:
                # Termination command
                response_text = "Goodbye."
                print("Voice thread: Stop command received. Exiting.")
                stop_event.set()  # signal the program to stop
            else:
                # General query – send to OpenAI GPT-3.5
                print("Voice thread: Querying GPT-3.5 for response...")
                # Append the user query to conversation history
                conversation.append({"role": "user", "content": user_text})
                try:
                    openai.api_key = "YOUR_OPENAI_API_KEY"  # Set your API key
                    gpt_resp = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=conversation
                    )
                    # Extract assistant response
                    response_text = gpt_resp['choices'][0]['message']['content'].strip()
                    # Add assistant reply to conversation history for context
                    conversation.append({"role": "assistant", "content": response_text})
                    print(f"Assistant (GPT response): {response_text}")
                except Exception as e:
                    # Handle API errors
                    response_text = "Sorry, I couldn't get an answer for that."
                    print(f"OpenAI API error: {e}")
            
            # Speak out the response text
            tts_engine.say(response_text)
            tts_engine.runAndWait()

            # If stop command was issued, break out after speaking
            if stop_event.is_set():
                break

    print("Voice thread: ending.")

# --- Face Recognition (Main Thread) ---
# Start the voice assistant thread (daemon=True so it exits when main exits)
voice_thread = threading.Thread(target=voice_assistant, daemon=True)
voice_thread.start()

# Open webcam video capture
print("Face thread: starting video capture...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot access the webcam.")
    stop_event.set()

# Loop over video frames
while not stop_event.is_set():
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to read frame from webcam.")
        break

    # Use DeepFace to detect faces in the frame
    try:
        faces = DeepFace.extract_faces(img_path=frame, detector_backend="opencv", enforce_detection=False)
    except Exception as e:
        print(f"DeepFace.extract_faces error: {e}")
        faces = []
    current_recognized = []  # names recognized in this frame

    # Process each detected face
    for face_obj in faces:
        face_img = face_obj["face"]        # cropped face image
        region = face_obj.get("facial_area", {})  # face bounding box
        # Get embedding for the face using our pre-loaded ArcFace model
        embedding = None
        try:
            rep = DeepFace.represent(img_path=face_img, model_name="ArcFace", model=model, enforce_detection=False)
            if rep and len(rep) > 0:
                embedding = rep[0]["embedding"]
        except Exception as e:
            print(f"Embedding extraction error: {e}")
        # Identify by comparing embedding with known embeddings
        name = "Unknown"
        if embedding is not None and known_embeddings:
            # Find closest known face by Euclidean distance
            best_match = None
            smallest_dist = float("inf")
            for person_name, ref_emb in known_embeddings.items():
                dist = np.linalg.norm(np.array(ref_emb) - np.array(embedding))
                if dist < smallest_dist:
                    smallest_dist = dist
                    best_match = person_name
            # Check if within threshold for a match
            if smallest_dist < 4.8:
                name = best_match
        current_recognized.append(name)

        # Draw bounding box and name label on the video frame (for visualization)
        if region:
            x, y, w, h = region.get("x",0), region.get("y",0), region.get("w",0), region.get("h",0)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    # Update the global recognized_names list (current people in frame)
    recognized_names[:] = current_recognized

    # Greeting logic: find newly recognized known people who haven't been greeted
    new_known = [nm for nm in current_recognized if nm != "Unknown" and nm not in greeted_names]
    if new_known:
        # Determine greeting message
        if len(new_known) > 1:
            greet_message = "Hi everyone!"
        else:
            greet_message = f"Hi {new_known[0]}!"
        print(f"Greeting: {greet_message}")
        # Mark these people as greeted
        for nm in new_known:
            greeted_names.add(nm)
        # Speak the greeting
        tts = pyttsx3.init()
        tts.say(greet_message)
        tts.runAndWait()
        # Signal that greeting happened, so voice thread can start listening
        if not greeting_event.is_set():
            greeting_event.set()

    # Display the video frame with drawings (press 'q' to quit)
    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Face thread: 'q' pressed, terminating.")
        stop_event.set()
        break

# Cleanup: release camera and close window
cap.release()
cv2.destroyAllWindows()
print("Face thread: Video capture stopped. Waiting for voice thread to finish...")
# Wait for voice thread to end
voice_thread.join(timeout=5.0)
print("Assistant has stopped. Exiting program.")
