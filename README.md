# Vision-based-object-detection-for-as-rs


# Bolt and Nut Classification and Size Estimation

This repository contains a Python script for detecting and classifying bolts and nuts using a YOLO model via a live camera feed. It also estimates the size of detected objects and allows users to capture annotated frames.

---

## Features

- **Real-time Object Detection**: Detects bolts and nuts using a pre-trained YOLO model.
- **Size Estimation**: Computes the dimensions of detected objects in millimeters based on a known reference.
- **Frame Capture**: Save annotated frames during the live feed by pressing a key.
- **User-Friendly Controls**:
  - Press **`q`** to quit the live feed.
  - Press **`c`** to capture and save the current annotated frame.

---

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/bolts-nuts-classifier.git
cd bolts-nuts-classifier
