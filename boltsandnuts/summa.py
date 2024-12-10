import os
from roboflow import Roboflow
from ultralytics import YOLO

# Initialize Roboflow with your API key
rf = Roboflow(api_key="DfniGOBmwy1M4Y97Upkg")

# Specify the project and version
project = rf.workspace("kim-sxidz").project("bolts-and-nuts-vhkyw")
version = project.version(8)

# Download dataset in YOLOv8 format
dataset = version.download("yolov8")

# Print the dataset location
print(f"Dataset downloaded to: {dataset.location}")

# Path to the dataset directory
dataset_path = dataset.location

# Training the YOLOv8 model
# Use YOLO class from ultralytics for training
model = YOLO("yolov8n.pt")  # Load a pre-trained YOLOv8 model (you can choose the size of the model)

# Define training parameters (You can modify these parameters)
train_params = {
    'data': os.path.join(dataset_path, 'data.yaml'),  # Dataset YAML file
    'epochs': 50,  # Number of training epochs
    'batch': 16,  # Batch size
    'imgsz': 640,  # Image size for training
    'weights': 'yolov8n.pt',  # Pre-trained weights (you can use a custom model if you prefer)
    'device': '0',  # Use GPU with index 0 (if available)
}

# Start the training process
results = model.train(**train_params)

# Once the model is trained, you can save the model
# The model is saved automatically to the 'runs/train/exp' folder, but you can save it to a custom path
model.save("best_model.pt")

# Print the results of the training
print(f"Training complete. Model saved as 'best_model.pt'.")
