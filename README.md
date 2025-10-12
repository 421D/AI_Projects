# AI_Projects


This repository contains AI-related projects from coursework, including **Machine Learning vs Deep Learning comparison** and **Deep Learning Fashion Classification**. Each project has its own folder with a detailed README, code, and report.
This repository contains my coursework and personal projects in machine learning and deep learning. The main focus is on image classification, model comparison, and deep learning experimentation.

## Projects Overview

### 1. ML vs DL Comparison
- **Folder:** `ML_vs_DL_Comparison`
- **Description:**  
  This project focuses on classifying aerial landscape images into **15 categories** (Airport, Beach, City, Forest, etc.) and compares **traditional machine learning methods** (LBP, SIFT + KNN/SVM/Random Forest/XGBoost) with **deep learning models** (ResNet-18 and EfficientNet-B0).  
- **Key Highlights:**  
  - Dataset: 12,000 images, 15 balanced categories  
  - Techniques: 5-fold cross-validation, data augmentation, transfer learning  
  - Insights: Deep learning outperforms traditional ML; EfficientNet-B0 is optimal for resource-constrained environments  
- **More Details:** See [`ML_vs_DL_Comparison/README.md`](ML_vs_DL_Comparison/README.md)


### 2. Fashion Classification (Deep Learning)
- **Folder:** `DL_Fashion_Classification`
- **Description:**  
  Classifies fashion items into **four categories**: Accessories, Bags, Clothing, Shoes using **pretrained CNNs** (MobileNetV2, ResNet18, ResNet50, EfficientNet-B0) on an **augmented dataset of 8,000 images**.  
- **Key Highlights:**  
  - Balanced dataset with data augmentation  
  - 5-fold cross-validation for model evaluation  
  - Insights: Lightweight models like MobileNetV2 and EfficientNet-B0 are ideal for efficiency; EfficientNet-B0 handles noise best  
- **More Details:** See [`DL_Fashion_Classification/README.md`](DL_Fashion_Classification/README.md)

