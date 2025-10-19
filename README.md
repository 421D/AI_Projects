# AI_Projects

This repository showcases selected coursework and personal projects in **machine learning (ML)** and **deep learning (DL)**, with a primary focus on **image classification, model performance comparison**, and **efficient deep learning deployment**.
The overall emphasis is on image classification, model evaluation, and transfer learning experiments.

## Projects Overview

### 1. ML vs DL: Aerial Image Classification | 传统 ML 与 DL 对比：遥感图像分类
- **Folder:** `ML_vs_DL_Comparison`
- **Goal:**  
Classify aerial images into **15 landscape categories** (e.g., Airport, Forest, City) using both **traditional ML techniques** (e.g., LBP, SIFT with KNN/SVM/Random Forest/XGBoost) and **modern deep learning architectures** (ResNet-18, EfficientNet-B0).
- **Key Highlights:**  
  - Dataset: Algorithms: LBP, SIFT + KNN / SVM / Random Forest / XGBoost
  - Deep Learning Models: ResNet-18, EfficientNet-B0
  - Dataset: 12,000 balanced images (15 categories, 256×256 px)
  - Methods: 5-fold cross-validation, data augmentation, transfer learning (ImageNet pretrained)
- Findings: Deep models achieve higher accuracy and robustness; EfficientNet-B0 offers the best trade-off between accuracy and efficiency.
- **For implementation details and performance reports, see:** See [`ML_vs_DL_Comparison/README.md`](ML_vs_DL_Comparison/README.md)


### 2. Fashion Classification (Deep Learning)
- **Folder:** `DL_Fashion_Classification`
- **Description:**  
  Classifies fashion items into **four categories**: Accessories, Bags, Clothing, Shoes using **pretrained CNNs** (MobileNetV2, ResNet18, ResNet50, EfficientNet-B0) on an **augmented dataset of 8,000 images**.  
- **Key Highlights:**  
  - Balanced dataset with data augmentation  
  - 5-fold cross-validation for model evaluation  
  - Insights: Lightweight models like MobileNetV2 and EfficientNet-B0 are ideal for efficiency; EfficientNet-B0 handles noise best  
- **More Details:** See [`DL_Fashion_Classification/README.md`](DL_Fashion_Classification/README.md)

