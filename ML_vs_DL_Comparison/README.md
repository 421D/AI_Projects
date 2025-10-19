# Aerial Landscape Image Classification

### Project Overview
This project aims to classify aerial landscape images into **15 categories** (e.g., Airport, Beach, City, Forest) and systematically compare the performance of **traditional machine learning methods** (LBP, SIFT + KNN/SVM/Random Forest/XGBoost) with **deep learning models** (ResNet-18 and EfficientNet-B0).
It evaluates feature-based and end-to-end approaches under a unified experimental setup.

Key techniques include:  
- **5-fold cross-validation** for stable and unbiased evaluation
- **Data augmentation** with flipping, rotation, colour jitter, and resized cropping
- **Evaluation metrics**: accuracy, precision, recall, F1-score, and confusion matrix

Deep learning models leverage **transfer learning** from ImageNet-pretrained weights for faster convergence and better generalisation.  



### Dataset
- **Source**: [Aerial_Landscapes on Kaggle](https://www.kaggle.com/datasets/balraj98/aerial-landscapes)  (dataset no longer available)
- **Composition**: 12,000 balanced images across 15 categories, each sized 256×256 pixels
- **Data loading**:  
  - ML methods: converted to NumPy arrays
  - CNN models: loaded using `torchvision.datasets.ImageFolder`  



### Libraries
- Python 3.10
- pandas, numpy, matplotlib, seaborn, tqdm, glob  
- OpenCV (`cv2`), scikit-image (`skimage`), scikit-learn (`sklearn`)  
- PyTorch (`torch`, `torchvision`); TensorFlow/Keras (for ML dataset preparation)



### Project Structure
```
├── dataset_split/
│ └── train/ # Pre-split training dataset
├── train_ResNet18.py
├── train_EfficientNet-B0.py
├── test_ResNet18.py
├── test_EfficientNet-B0.py
├── split_dataset.py # Script to create reduced training dataset
├── model_fold{n}.pth # Saved models per fold
├── fold{n}_report.txt # Classification report per fold
└── README.md
```




### Traditional ML Methods
**Feature Extraction**:  
- **LBP (Local Binary Patterns)**: generates a 10-bin histogram per image
- **SIFT (Scale-Invariant Feature Transform)**: extracts 50 keypoints × 128-dim vectors forming a 6,400-dim representation

**Classifiers**:  
- K-Nearest Neighbours (basic & weighted)  
- Random Forest (basic & weighted)  
- SVM (basic & weighted)  
- XGBoost (for imbalanced experiments)  

**Evaluation Metrics**:  
- Training/prediction time  
- Accuracy  
- Precision, Recall, F1-score  
- Confusion matrix  

**Key Observations**:  
- LBP + Random Forest achieved the best performance among ML methods.
- Weighted models improved accuracy by approximately 2–4%.
- XGBoost reached 47.69% accuracy on an imbalanced dataset.


### Deep Learning Methods
**Models**:  
- **ResNet-18** (11.7M parameters)  
- **EfficientNet-B0** (5.3M parameters, more efficient)  

**Training Setup**:  
Input size 224×224, 5-fold cross-validation, Adam optimizer (lr = 1e-4), batch size 32, 10 epochs,
with data augmentation (horizontal flip, ±15° rotation, colour jitter, random resized crop).

**Performance**:  
| Model          | Average Accuracy | Precision | Recall | F1-score |
|----------------|----------------|-----------|--------|----------|
| ResNet-18      | 96.93%         | 96.95%    | 96.94% | 96.92%   |
| EfficientNet-B0| 97.99%         | 98.01%    | 97.99% | 97.98%   |

- EfficientNet-B0 consistently outperformed ResNet-18 across all folds.
- Augmentation and transfer learning significantly improved generalisation, and confusion matrices indicated near-perfect separation across major categories.



### Key Insights
- **Deep learning > traditional ML** in accuracy and robustness  
- **EfficientNet-B0** is optimal for resource-constrained environments  
- Weighted models improve performance on imbalanced datasets  
- Strategic **feature selection** and **model choice** balance efficiency and accuracy  
- Data augmentation and transfer learning accelerate convergence and improve generalisation  



### How to Run
1. Install required libraries:  
```
conda create -n torch310 python=3.10 -y
conda activate torch310
pip install torch torchvision scikit-learn scikit-image opencv-python pandas numpy matplotlib seaborn tqdm
```

2. Download the dataset from Kaggle and place it in dataset_split/train/
3. Split dataset (optional reduced version): python split_dataset.py
4. Train models:  python train_ResNet18.py    python train_EfficientNet-B0.py
5. Test models:python test_ResNet18.py
python test_EfficientNet-B0.py

6. Check results:
  Fold reports: fold{n}_report.txt
  Saved models: model_fold{n}.pth
  Confusion matrices & metrics saved as PNG/TXT files
