# Hotspot Detection for Drone Payload Delivery

This project implements a YOLOv8-based computer vision system to detect concentric circle "hotspot" targets for drone payload drop precision. It includes dataset generation, model training scripts, and drone command integration.

## Features

- Dataset generation and labeling scripts for hotspot images
- YOLOv8 training configuration and pre-trained models
- Real-time hotspot detection using webcam input
- Command modules to control drone behavior based on detection
- Training results visualization (confusion matrix, F1 score, etc.)

## Repository Structure

```bash
Hotspot/
├── generate_dataset.py # Data collection and labeling script
├── generate_yolo_dataset.py # Dataset formatting for YOLO
├── split_dataset.py # Splits dataset into train/val/test
├── hotspot.yaml # YOLO training config file
├── hotspot_command.py # Main detection and control logic
├── send_commands.py # Drone command sender
├── send_command_cv.py # OpenCV variant of command sender
├── send_command_advance.py # Advanced command sender
├── yolo11n.pt # YOLO model weights
├── yolov8n.pt # YOLOv8 nano model weights
├── hotspot_runs/ # Training results (plots, logs)
│ └── train/
│ ├── confusion_matrix.png
│ ├── F1_curve.png
│ └── ...
└── yolo_hotspot_dataset/ # Dataset (ignored in repo)
├── images/
└── labels/
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/kabhishek3001/Hotspot-detection.git
cd Hotspot-detection
```

Install dependencies (recommended to use a virtual environment):
```bash
pip install -r requirements.txt
```

Note: You may need OpenCV, PyTorch, and Ultralytics YOLO libraries.

Usage
Generate Dataset: Use generate_dataset.py and generate_yolo_dataset.py to create labeled data.

Split Dataset: Run split_dataset.py to divide data into train/val/test sets.

Train Model: Use the YOLO training scripts with hotspot.yaml config.

Detect Hotspot: Run hotspot_command.py for real-time detection and drone control.

Contributing
Contributions and suggestions are welcome! Feel free to open issues or submit pull requests.

License
This project is licensed under the MIT License.

Made by Abhishek Kamble 
