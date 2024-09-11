# Traffic-Road-Object-Detection
This project uses deep learning to perform object detection and classification on images and videos using YOLOv8 in real-time. Use DVC for version control, to ensure the Continuous Integration (CI) pipeline for model training, testing, and deployment is automated
the data used from kaggel [Traffic Road Object Detection Polish Dataset](https://www.kaggle.com/datasets/mikoajkoek/traffic-road-object-detection-polish-12k)

# Build Tool
- PyTorch
- YOLOv8
- DVC (Data Version Control)
- OpenCV

# Installation
- install anacodna
- create a conda env
```
conda create -n your_env_name
```
- activate your env
```
conda activate your_env_name
```
- install the requirments
```
pip instals -r requiremnets.txt
```
- run dvc repro
```
dvc repro
```

# Model Results 
## Results
![Results](/best/YOLOv8n_custom_1/results.png)
## Confusion Matrix
![Confusion Matrix](/best/YOLOv8n_custom_1/confusion_matrix_normalized.png)
## F1 Curve
![F1 score](/best/YOLOv8n_custom_1/F1_curve.png)
## PR Curve
![PR Curve](/best/YOLOv8n_custom_1/PR_curve.png)
## R Curve
![R Curve](/best/YOLOv8n_custom_1/R_curve.png)