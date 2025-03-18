import os
import cv2
import pickle
from deepface import DeepFace

KNOWN_FACES_DIR = "known_faces"
ENCODINGS_FILE = "encodings.pickle"

known_face_encodings = {}
print("Encoding faces...")

# Loop through images
for filename in os.listdir(KNOWN_FACES_DIR):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(KNOWN_FACES_DIR, filename)

        try:
            encoding = DeepFace.represent(image_path, model_name="Facenet")[0]["embedding"]
            known_face_encodings[os.path.splitext(filename)[0]] = encoding
        except Exception as e:
            print(f"Error encoding {filename}: {e}")

# Save encodings
with open(ENCODINGS_FILE, "wb") as f:
    pickle.dump(known_face_encodings, f)

print("Face encodings saved!")
