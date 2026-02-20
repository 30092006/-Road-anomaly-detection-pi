import cv2
import os
from ultralytics import YOLO

# 1. SETUP PATHS
model_path = "best_int8.tflite"  # Path to your converted TFLite file
output_path = "/home/pi/Desktop/detections"  # Change this to your desired location

if not os.path.exists(output_path):
    os.makedirs(output_path)

# 2. LOAD MODEL
# YOLOv8 handles TFLite automatically
model = YOLO(model_path, task='detect')

# 3. INITIALIZE CAMERA
cap = cv2.VideoCapture(0)  # '0' is usually the default USB or Pi camera

print("Starting inference... Press 'q' to stop.")

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 4. RUN INFERENCE
    # imgsz=320 must match the size you used during conversion for best speed
    results = model.predict(frame, imgsz=320, conf=0.25, verbose=False)

    # 5. PROCESS & SAVE RESULTS
    # results[0].plot() draws the boxes and labels on the image
    annotated_frame = results[0].plot()
    
    # Save the frame to your specific location
    file_name = os.path.join(output_path, f"result_{frame_count}.jpg")
    cv2.imwrite(file_name, annotated_frame)
    
    # Optional: Show the live feed (comment out to save CPU power)
    cv2.imshow("YOLOv8 Pi4", annotated_frame)

    frame_count += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
