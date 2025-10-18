# 🍽️ Machine Learning-Based Table Queueing System

A **computer vision–based queue management system** designed to detect and classify **table occupancy** in eateries using **YOLOv4 (You Only Look Once)** for real-time object detection.  
The system identifies available and occupied tables to help automate seat allocation and improve customer flow management.

---

## 🧠 Project Overview

This project applies **machine learning and image processing** to analyze live video feeds from a restaurant or cafeteria.  
Using the **YOLOv4** deep learning model, the system detects objects such as `dining tables`, `chairs`, and `people` to determine whether a table is **available** or **occupied**.

It was built for prototype purposes using a sample video (eatery5.mp4), but can also be extended to use live camera feeds (e.g., CCTV or webcam) with minor modifications.

---

## 🧩 Key Features
- 🧠 **YOLOv4-based detection** of tables, chairs, and people  
- 🎯 **Occupancy classification** based on detected objects in table regions  
- 📹 **Real-time video inference** using OpenCV  
- ⚙️ **Object tracking** for consistent detection across frames  
- 💡 Works under various **lighting and seating conditions**

---

## 🧰 Technologies Used

| Category             | Tools / Libraries                               |
| -------------------- | ----------------------------------------------- |
| Programming Language | Python                                          |
| Machine Learning     | YOLOv4 (Darknet model)                          |
| Framework            | OpenCV DNN                                      |
| Object Tracking      | Custom centroid-based tracker                   |
| Dependencies         | NumPy, OpenCV, time, math                       |
| IDE                  | Visual Studio Code / PyCharm / Jupyter Notebook |


```

## 📁 Project Structure
Machine-Learning-Table-Queueing-System/
│
├── main.py # Main script for video detection and classification
├── object_detection.py # YOLOv4 model setup and inference
├── object_tracking.py # Tracking logic for consistent detection
│
├── dnn_model/
│ ├── yolov4.cfg # YOLOv4 configuration file
│ ├── yolov4.weights # Pretrained weights file
│ └── classes.txt # COCO dataset class names
│
├── videos/
│ └── eatery5.mp4 # Sample test video (not included)
│
└── README.md
```

## 🧩 How It Works

1. **Load YOLOv4 Model**  
   The system loads pretrained YOLOv4 weights and configuration using OpenCV’s DNN module.

2. **Object Detection**  
   Each frame from the video is analyzed to detect objects (tables, chairs, people).

3. **Region Analysis**  
   Detected `diningtable` regions are cropped, and additional detections are checked within those regions to determine occupancy.

4. **Classification Logic**  
   - If `person`, `chair`, or `bowl` objects are detected within the table’s region → **Occupied**  
   - Otherwise → **Available**

5. **Display Output**  
   The system draws bounding boxes and labels (“Available” or “Occupied”) around tables in real time.

---

## ⚙️ Installation & Setup

### 1️⃣ Clone this Repository
```bash
git clone https://github.com/<your-username>/machine-learning-table-queueing-system.git
cd machine-learning-table-queueing-system

2️⃣ Install Dependencies
Make sure Python is installed, then run:

bash
Copy code
pip install opencv-python numpy

3️⃣ Download YOLOv4 Model Files
You’ll need:

yolov4.weights
yolov4.cfg
classes.txt

You can download YOLOv4 pretrained weights from:
https://pjreddie.com/darknet/yolo/

Place them inside the dnn_model/ folder.

▶️ Running the System
🧩 Option 1: Use a Sample Video (Prototype Mode)
python main.py

By default, the code uses the eatery5.mp4 file as input.

🧩 Option 2: Use a Custom Video
python main.py --video path/to/your/video.mp4

🧩 Option 3: Use a Live Camera Feed (Optional Enhancement)

If you want to use a webcam or CCTV instead of a saved video, modify this line in main.py:

cap = cv2.VideoCapture(0)  # Use 0 for default webcam

⚠️ Only a few minor code edits are needed — the detection logic remains the same.

🔬 Example Applications

Restaurant or cafeteria queue management

Smart cafeteria monitoring systems

Real-time occupancy detection for space optimization

Integration with booking or self-seating systems

💡 Future Improvements

Replace YOLOv4 with YOLOv8 for faster performance

Add multi-camera support for large spaces

Integrate with a web dashboard for live seat status

Deploy model using TensorRT or ONNX for edge devices (e.g., Jetson Nano)

⚠️ Notes

If the system runs slowly on your computer:

Reduce frame size in main.py by adjusting:

resize_scale = 0.5  # Lower value for faster processing

Use GPU acceleration (CUDA) if available in OpenCV.

📜 License

This project is open-source and intended for educational and research purposes under the MIT License.

👨‍💻 Developer: Ralph Buenaventura
🎓 Bachelor of Science in Computer Engineering
📍 Philippines
🔗 https://github.com/raaalphhh
