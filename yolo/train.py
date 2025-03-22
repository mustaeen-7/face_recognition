import cv2
import os

# Set up the database folder
db_path = "faces_db"
os.makedirs(db_path, exist_ok=True)

name = input("Enter your name: ").strip().replace(" ", "_")

cap = cv2.VideoCapture(0)

for i in range(5):  # Capture 5 images at different angles
    input(f"Look in different direction & press Enter ({i+1}/5)")
    ret, frame = cap.read()
    if not ret:
        break
    img_path = os.path.join(db_path, f"{name}_{i}.jpg")
    cv2.imwrite(img_path, frame)
    print(f"Saved: {img_path}")

cap.release()
cv2.destroyAllWindows()
