# Fashion Classifier
---
For further details, please refer to the `notebook.ipynb`
---
### Project Overview
This project focuses on the classification of fashion images into **four categories** — Accessories, Bags, Clothing, and Shoes.
It employs **deep learning-based image classification** using **pretrained convolutional neural networks (CNNs)**, including MobileNetV2, ResNet18, ResNet50, and EfficientNet-B0.
The dataset consists of **8,000 balanced and augmented images**, designed to evaluate accuracy, generalisation, and robustness across different network architectures.


Key objectives:  
- Automate product labelling for online retail applications
- Achieve high classification accuracy with limited computational resources
- Handle intra-class variations and inter-class similarities effectively



### Dataset
- Original datasets: LAT & AAT  
- **Data Cleaning**: manually corrected mislabelled samples
- **Data Enhancement**:  
  - Balancing to 2,000 images per category (total 8,000)  
  - Automatic labelling based on filename patterns  
- **Augmentation Techniques**: rotation, flipping, and scaling to improve model robustness



### Libraries & Tools
- Python 3.x 
- PyTorch (`torch`, `torchvision`)  
- TensorFlow/Keras (data loading & preprocessing)  
- scikit-learn (metrics & evaluation)  
- pandas, numpy, matplotlib, seaborn  



### Model Overview
The project evaluates four pretrained CNN architectures differing in complexity and parameter

| Model           | Parameters | Notes |
|-----------------|------------|-------|
| MobileNetV2     | 3.5M       | Lightweight architecture, fast inference |
| ResNet18        | 11.7M      | Balanced trade-off between speed and accuracy |
| ResNet50        | 25.6M      | Deeper model with stronger feature extraction |
| EfficientNet-B0 | 5.3M       | Compound scaling, optimized for accuracy-efficiency ratio |

**Training Setup**:  
- 5-fold cross-validation for reliable evaluation
- Adam optimiser with learning rate scheduling
- Early stopping to prevent overfitting
- Data augmentation applied during training


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
- All models achieve ≈99% accuracy on clean data.
- EfficientNet-B0 maintains the highest performance under noisy conditions (93% accuracy, 0.990 mAP).
- ResNet50 shows overfitting on smaller datasets.
- MobileNetV2 and EfficientNet-B0 are ideal for deployment in low-resource systems.



### Key Insights
1. **Model Selection**: ResNet18 balances performance and efficiency; ResNet50 achieves peak accuracy when resources are sufficient.
2. **Data Augmentation**: Essential for improving robustness and mitigating class imbalance.
3. **Noise Sensitivity**: Deeper models with higher capacity are prone to overfitting smaller datasets.
4. **EfficientNet-B0**: Provides the optimal trade-off between accuracy, generalisation, and inference speed.



### How to Run
1. Install required libraries:
```
pip install torch torchvision tensorflow scikit-learn pandas numpy matplotlib seaborn
```
2. Place the augmented dataset under dataset_augmented/

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


