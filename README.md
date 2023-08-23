# FaceScan

FaceScan is a Python application for capturing and recognizing faces using OpenCV. It allows you to add new faces to the dataset and perform face recognition using the LBPH (Local Binary Patterns Histograms) algorithm.

## Features

- Capture and save face images to the dataset.
- Perform face recognition based on the LBPH algorithm.

## Prerequisites

- Python 3.x
- OpenCV (`pip install opencv-python`)
- NumPy (`pip install numpy`)

## Usage

1. Run the `main.py` script to start the application.
2. Choose an option:
   - Press `1` to add a new face.
   - Press `2` to detect and recognize faces.
3. If adding a new face:
   - Enter the person's name.
   - Follow the on-screen instructions to capture face images.
4. If detecting and recognizing faces:
   - The application will use the LBPH algorithm to recognize faces in real-time.

## Configuration

- The Haarcascades face detection model is loaded from OpenCV's data directory.
- Face images are saved to the `data` directory under the person's name subdirectories.

## Contributors

- [Hamdi Mokni](https://github.com/Mk-1000)

## License
