# Fashion Classifier

### Project Overview
This project aims to classify fashion items into **four main categories**:  
- Accessories  
- Bags  
- Clothing  
- Shoes  

Using **deep learning-based image classification**, we leverage **pretrained CNNs** (MobileNetV2, ResNet18, ResNet50, EfficientNet-B0) on a **balanced, augmented dataset** of 8,000 images.  

Key objectives:  
- Automate fashion item labelling for online retail  
- Achieve high accuracy with limited computational resources  
- Handle intra-class variation, inter-class similarity, and class imbalance  



### Dataset
- Original datasets: LAT & AAT  
- **Data Cleaning**: Misclassified samples manually corrected  
- **Data Enhancement**:  
  - Balancing to 2,000 images per category (total 8,000)  
  - Automatic labelling based on filename patterns  
- **Augmentation Techniques**: rotation, flipping, and scaling to improve robustness  



### Libraries & Tools
- Python 3.x  
- PyTorch (`torch`, `torchvision`)  
- TensorFlow/Keras (data loading & preprocessing)  
- scikit-learn (metrics & evaluation)  
- pandas, numpy, matplotlib, seaborn  



### Model Overview

| Model           | Parameters | Notes |
|-----------------|------------|-------|
| MobileNetV2     | 3.5M       | Lightweight, fast inference |
| ResNet18        | 11.7M      | Good balance of accuracy and efficiency |
| ResNet50        | 25.6M      | Stronger feature extraction, slower training |
| EfficientNet-B0 | 5.3M       | Compound scaling, high accuracy and efficiency |

**Training Setup**:  
- 5-fold cross-validation  
- Adam optimizer + learning rate scheduler  
- Early stopping to prevent overfitting  
- Data augmentation during training  



### Results

#### 5-Fold Cross-Validation (Validation Loss)
| Model           | Avg Validation Loss | Best Fold Loss | Epochs |
|-----------------|------------------|----------------|--------|
| MobileNetV2     | 0.0686           | 0.0610         | 20     |
| ResNet18        | -                | -              | -      |
| ResNet50        | -                | -              | -      |
| EfficientNet-B0 | 0.0730           | 0.0660         | 10     |

#### Test Results (Stratified)
| Model           | Accuracy (%) | F1 Score | mAP  |
|-----------------|-------------|----------|------|
| MobileNetV2     | 99.60       | 0.996    | 1.000|
| ResNet18        | 98.80       | 0.988    | 1.000|
| ResNet50        | 99.80       | 0.998    | 1.000|
| EfficientNet-B0 | 99.80       | 0.998    | 1.000|

#### Test Results with Gaussian Noise
| Model           | Accuracy (%) | F1 Score | mAP  |
|-----------------|-------------|----------|------|
| MobileNetV2     | 92.40       | 0.924    | 0.976|
| ResNet18        | 91.40       | 0.914    | 0.986|
| ResNet50        | 89.00       | 0.890    | 0.980|
| EfficientNet-B0 | 93.00       | 0.930    | 0.990|

**Key Observations**:  
- All models achieve near-perfect accuracy without noise (≈99%)  
- EfficientNet-B0 outperforms others under noise (93% accuracy, 0.990 mAP)  
- ResNet50 underperforms ResNet18 with small datasets due to overfitting  
- Lightweight models like MobileNetV2 and EfficientNet-B0 are ideal for resource-constrained environments  



### Key Insights
1. **Model Selection**: Choose ResNet18 for efficiency, ResNet50 for maximum accuracy if resources permit.  
2. **Data Augmentation**: Essential for handling noise and class imbalance.  
3. **Noise Sensitivity**: Deeper models with more parameters are prone to overfitting small datasets.  
4. **EfficientNet-B0**: Best trade-off between accuracy, generalisation, and efficiency.  



### How to Run
1. Install required libraries:
```
pip install torch torchvision tensorflow scikit-learn pandas numpy matplotlib seaborn
```
2. Download the augmented dataset into dataset_augmented/

3. Train models with 5-fold cross-validation:
```
python train_MobileNetV2.py
python train_ResNet18.py
python train_ResNet50.py
python train_EfficientNet-B0.py
```
4. Test models and generate reports:
```
python test_models.py
```

5. Check results:
Fold reports: fold{n}_report.txt
Saved models: model_fold{n}.pth
Confusion matrices & metrics available for analysis


