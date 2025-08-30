import cv2, time
import numpy as np
import pandas as pd
from ultralytics import YOLO
import os

# Ensure output folders exist
os.makedirs("output", exist_ok=True)
os.makedirs("data", exist_ok=True)

model = YOLO("yolov8n.pt")  # choose nano for speed-friendly

# Load video
cap = cv2.VideoCapture("videos/sample.mp4")
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter("output/result.mp4", cv2.VideoWriter_fourcc(*"XVID"), fps, (width, height))

# Data store
records = []

frame_id = 0
max_frames = 50 # process only 50 frames

while True:
    ret, frame = cap.read()
    if not ret or frame_id >= max_frames:
        break

    frame_id += 1
    results = model(frame)[0]

    for box in results.boxes:
        cls = int(box.cls[0])
        if cls in [2, 3, 5, 7]:  # car, motorbike, bus, truck
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            records.append([frame_id, cls, x1, y1, x2, y2])

    out.write(frame)

# Write to CSV
df = pd.DataFrame(records, columns=["frame", "class", "x1","y1","x2","y2"])
df.to_csv("data/vehicle_counts.csv", index=False)

cap.release()
out.release()
cv2.destroyAllWindows()
print("Processing complete — video and CSV saved (first 50 frames only).")
