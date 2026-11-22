# ARM Bharat AI-SoC
### Offline AI-Powered Crop Disease Detection and Advisory System

## Overview

ARM Bharat AI-SoC is an offline AI-powered crop disease detection and advisory system designed to assist farmers in identifying plant diseases through image analysis.

The system combines deep learning-based image classification with a locally running Large Language Model (LLM) to provide disease identification, relevant information, and agricultural advisory without depending on internet connectivity during operation.

## Problem Statement

Farmers may face difficulties identifying crop diseases at an early stage due to limited access to agricultural experts, connectivity limitations, and delayed diagnosis.

This project aims to develop an accessible offline solution that analyzes plant images, identifies possible diseases, and provides useful disease-related information and preventive guidance.

## Proposed Solution

The proposed system uses a CNN-based image classification model to identify crop diseases from plant images. The classification output is then passed to a locally running Llama language model to generate understandable disease-related explanations and advisory.

The system is designed to operate offline, making it suitable for agricultural environments with limited internet connectivity.

## System Workflow

1. Capture or input a plant image.
2. Preprocess the image.
3. Analyze the image using a CNN model.
4. Predict the crop disease or healthy condition.
5. Pass the prediction and relevant disease information to the local Llama model.
6. Generate disease explanation, precautions, and advisory.
7. Display the results to the user.

## Key Features

- Offline crop disease detection
- CNN-based plant image classification
- Local Llama-based advisory generation
- Disease information and preventive guidance
- No dependence on cloud AI services during operation
- Designed for FPGA-based hardware integration
- Intended for low-connectivity agricultural environments

## AI Architecture

### 1. CNN – Disease Detection

A Convolutional Neural Network (CNN) built using TensorFlow is used to analyze plant leaf images and classify them into crop disease or healthy categories.

### 2. Llama – Advisory Generation

Llama, running locally through Ollama, is used to generate farmer-friendly explanations and advisory based on the disease prediction and relevant disease information.

The local LLM approach is intended to avoid dependence on cloud-based AI services during operation.

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
CNN Model (TensorFlow)  
↓  
Disease Prediction  
↓  
Local Llama Model (Ollama)  
↓  
Disease Explanation & Farmer Advisory

## Technologies Used

- Python
- TensorFlow
- CNN (Convolutional Neural Network)
- Ollama
- Llama LLM
- Image Processing
- FPGA Hardware Integration

## Offline Operation

The proposed system uses locally stored dataset/model resources and a locally running Llama model through Ollama.

Disease classification and advisory generation are designed to work without sending plant images or user requests to cloud AI services.

## Hardware Implementation

The project aims to integrate the AI-based crop disease detection pipeline with an FPGA-based hardware platform.

The hardware implementation focuses on exploring efficient image processing and AI inference in an offline agricultural application.

## Project Status

- [x] Dataset collection and organization
- [x] Initial software prototype
- [x] CNN disease classification pipeline development
- [x] Local Llama advisory integration
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

An offline crop disease detection and advisory prototype capable of analyzing plant images using a CNN and generating understandable disease-related guidance using a locally running Llama model, with FPGA integration as the intended hardware implementation.

## Disclaimer

This system is an experimental prototype intended to support crop disease identification. Its predictions and advisory should not replace professional agricultural diagnosis.

## Project

**ARM Bharat AI-SoC**

AI-powered offline crop disease detection and advisory for farmers.