🚦 Traffic Flow Analysis with YOLOv8

This project uses YOLOv8 (You Only Look Once) for detecting and counting vehicles in traffic videos. The system processes video frames, identifies cars, buses, trucks, and motorbikes, and generates useful logs for traffic analysis.

📺 Traffic Video
https://www.youtube.com/watch?v=MNn9qKG2UFI
👉 Use this video on youtube

📂 Dataset Explanation

The code automatically creates a CSV file (vehicle_counts.csv) inside the data/ folder.

Each row in the CSV represents a vehicle detected in a frame.

The dataset includes:

Frame	Class	x1	y1	x2	y2
1	2 (Car)	50	100	200	300
1	3 (Motorbike)	300	120	400	250

Frame → Which video frame the object was detected in.

Class → Vehicle type (2 = Car, 3 = Motorbike, 5 = Bus, 7 = Truck).

x1, y1, x2, y2 → Coordinates of the bounding box around the detected vehicle.

This dataset can later be used for:

🚗 Traffic density analysis

⏱ Average waiting time at signals

📊 Vehicle type distribution

⚙️ How It Works

Load the input video (videos/sample.mp4).

Run YOLOv8 object detection on each frame.

Detect vehicles (car, motorbike, bus, truck).

Save the annotated video in output/result.mp4.

Store vehicle logs in data/vehicle_counts.csv.
