import sys
import os
# Ensure the yolov8 directory is in the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'yolov8'))
from yolov8.ultralytics.utils import ASSETS
from yolov8.ultralytics import YOLO
from yolov8.ultralytics.models.yolo.detect.predict import DetectionPredictor
import torch
import numpy as np
import cv2
from time import time
import argparse

        
        

def main():
    # Load your trained YOLOv8 model
    pred_model= f'best/best_latest.pt' if os.path.exists(f'best/best_latest.pt') else  "yolov8n.pt"
    model = YOLO(pred_model)  # Update with your model's path
    # print(dir(model))
    # results = model(source="0", stream=True)  # return a generator of Results objects
    
    # # Process results generator
    # for result in results:
    #     boxes = result.boxes  # Boxes object for bounding box outputs


    # Open a connection to the webcam
    cap = cv2.VideoCapture(1)  # 0 is typically the default camera

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture image.")
            break

        # Run YOLOv8 prediction on the frame
        results = model(frame, show=False)  # Set show=True if you want YOLO to display results automatically

        # Draw bounding boxes and labels on the frame
        annotated_frame = results[0].plot()  # YOLOv8's built-in function to plot boxes/labels on the image

        # Display the resulting frame
        cv2.imshow('YOLOv8 Webcam', annotated_frame)

        # Break the loop if the user presses the 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()
if __name__ == '__main__':
    main()








"""
    A class extending the BasePredictor class for prediction based on a detection model.

    Example:
        ```python
        from ultralytics.utils import ASSETS
        from ultralytics.models.yolo.detect import DetectionPredictor

        args = dict(model="yolov8n.pt", source=ASSETS)
        predictor = DetectionPredictor(overrides=args)
        predictor.predict_cli()
        ```
"""




# class ObjectDetection:
#     def __init__(self, capture_index) -> None:
#         self.capture_index = capture_index
#         self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
#         print("using device: ",self.device)


#     def __call__(self):
#             cap = cv2.VideoCapture(self.capture_index)
#             # assert cap.isOpened()
#             if not cap.isOpened():
#                 print("Cannot open camera")
#                 exit()

#             while True:
#                 # Capture frame-by-frame
#                 ret, frame = cap.read()

#                 # If the frame is read correctly, ret will be True
#                 if not ret:
#                     print("Can't receive frame (stream end?). Exiting ...")
#                     break

#                 # Display the resulting frame
#                 cv2.imshow('Camera', frame)

#                 # Press 'q' to exit the loop
#                 if cv2.waitKey(1) == ord('q'):
#                     break

#             # Release the camera and close all windows
#             cap.release()
#             cv2.destroyAllWindows()