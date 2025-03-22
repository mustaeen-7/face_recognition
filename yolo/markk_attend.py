import cv2
import pandas as pd
from datetime import datetime
from deepface import DeepFace
from ultralytics import YOLO

# Load pre-trained YOLO model for face detection
yolo_model = YOLO("yolov8n-face.pt")  # You can train your own model if needed

# Load webcam
cap = cv2.VideoCapture(0)

# Load attendance CSV file
attendance_file = "attendance.csv"

def mark_attendance(name):
    df = pd.read_csv(attendance_file) if attendance_file else pd.DataFrame(columns=["Name", "Time"])
    
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_entry = pd.DataFrame({"Name": [name], "Time": [now]})
    
    df = pd.concat([df, new_entry], ignore_index=True)
    df.to_csv(attendance_file, index=False)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect faces using YOLO
    results = yolo_model(frame)

    for box in results.xyxy[0]:  # Iterate over detected faces
        x1, y1, x2, y2 = map(int, box[:4])
        face_crop = frame[y1:y2, x1:x2]

        # Recognize the face
        try:
            result = DeepFace.find(img_path=face_crop, db_path="faces_db/", model_name="ArcFace")

            if len(result) > 0:
                person_name = result[0]["identity"].split("/")[-1].split(".")[0]
                mark_attendance(person_name)

                # Draw rectangle and name
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, person_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        except Exception as e:
            print(f"Recognition error: {e}")

    cv2.imshow("Face Recognition Attendance", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
