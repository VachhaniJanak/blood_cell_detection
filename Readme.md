# Blood Cell Detection Using Faster R-CNN

This repository contains the implementation of a blood cell detection model using Faster R-CNN. The model is designed to detect and classify three types of blood cells in microscopic images.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Usage](#usage)
- [Training the Model](#training-the-model)

## Overview

The goal of this project is to accurately detect various types of blood cells using the Faster R-CNN architecture. The Faster R-CNN model provides fast and accurate detection by combining region proposal networks with object detection.

## Dataset

Dataset found inside dataset directory.

The dataset consists of:
- Microscopic images of blood cells.
- Annotations in the form of bounding boxes for various blood cell types.

## Model Architecture

The model is based on the Faster R-CNN architecture, which consists of:
1. **Backbone**: A CNN (e.g., ResNet-50) used for feature extraction.
2. **Region Proposal Network (RPN)**: Proposes candidate object bounding boxes.
3. **ROI Pooling**: Extracts features from the proposed regions.
4. **Bounding Box Regression and Classification**: Predicts the class and refines bounding boxes.

## Usage

### Running the Model
You can use the pre-trained model for inference on new images:

```bash
python detect.py "/path/to/source_image.jpg" "/path/to/destination_image.jpg"
```

### Training the Model

Whole tranning process found inside "blood-cell-detection.ipynb":

### Accuracy Plot
Here’s a plot of the accuracy over epochs during training:

![Accuracy Plot](/imgs/plot.png) <!-- Placeholder for accuracy plot -->

### Example Detections

Below are some example detections made by the model:

![Detection Example 1](/imgs/output.png)  <!-- Placeholder for image -->
![Detection Example 2](/imgs/output1.png)  <!-- Placeholder for image -->

