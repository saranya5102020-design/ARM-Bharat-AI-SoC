# ARM Bharat AI-SoC
### Offline AI-Powered Crop Disease Detection and Advisory System

## Overview

ARM Bharat AI-SoC is an offline AI-powered crop disease detection and advisory system designed to assist farmers in identifying plant diseases through image analysis.

The system aims to integrate artificial intelligence, image processing, and FPGA-based hardware acceleration to provide disease identification and agricultural advisory support without depending on an internet connection.

## Problem Statement

Farmers may face difficulties identifying crop diseases at an early stage due to limited access to agricultural experts, connectivity limitations, and delayed diagnosis.

This project aims to develop an accessible, offline solution that analyzes plant images and provides disease-related information and preventive guidance.

## Proposed Solution

The system uses a deep learning-based image classification model to analyze crop images and identify possible diseases.

### Workflow

1. Capture or input a plant image.
2. Preprocess the image.
3. Process it using a trained CNN model.
4. Identify the predicted crop disease or healthy condition.
5. Provide relevant disease information and advisory.
6. Integrate the processing pipeline with FPGA hardware.

## Key Features

- Offline crop disease detection
- Deep learning-based image classification
- Multiple crop and disease categories
- Disease-related information and preventive advisory
- Designed for FPGA-based hardware integration
- Intended for low-connectivity agricultural environments

## Dataset

The model development uses the PlantVillage dataset.

| Dataset Information | Details |
|---|---|
| Total Images | 54,305 |
| Classes | 38 |
| Dataset Type | Plant leaf images |
| Image Format | JPG |
| Dataset Usage | Model training and evaluation |

The dataset is maintained locally and is not included in this GitHub repository due to its large size.

### Dataset Source

PlantVillage Dataset:

https://github.com/spMohanty/PlantVillage-Dataset

## Project Workflow

Plant Image  
↓  
Image Preprocessing  
↓  
CNN Classification  
↓  
Disease Prediction  
↓  
Advisory Output

## Technologies Used

- Python
- TensorFlow
- CNN (Convolutional Neural Network)
- Image Processing
- FPGA Hardware Integration

## Hardware Implementation

The project aims to integrate the crop disease detection pipeline with an FPGA-based hardware platform.

The hardware implementation focuses on exploring efficient processing of image data and AI inference in an offline environment.

## Project Status

- [x] Dataset collection and organization
- [x] Initial software prototype
- [x] Disease classification pipeline development
- [x] Advisory generation module
- [ ] FPGA integration
- [ ] Hardware testing and validation
- [ ] Complete offline hardware demonstration

## Repository Structure

    ARM-Bharat-AI-SoC/
    │
    ├── model/
    │   └── CNN model files
    │
    ├── src/
    │   └── Disease detection and advisory code
    │
    ├── README.md
    └── requirements.txt

## Expected Outcome

A prototype capable of identifying crop diseases from plant images and generating relevant advisory information in an offline environment, with FPGA integration as the intended hardware implementation.

## Disclaimer

This system is an experimental prototype intended to support crop disease identification. Its predictions should not replace professional agricultural diagnosis.

## Team Project

**ARM Bharat AI-SoC**

AI-powered offline crop disease detection and advisory for farmers.