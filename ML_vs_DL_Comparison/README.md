# 9517
# Aerial Landscape Image Classification using machine learning techniques and Convolutional Neural Networks

### 📌 Overview
This project aims to classify aerial images into 15 landscape categories (e.g., Airport, Beach, City, Forest, etc.). Machine learning covers data loading and preprocessing, feature extraction using **LBP** and **SIFT**, and model training and evaluation with various classifiers. using convolutional neural networks using **EfficientNet-B0** and **ResNet18**. We applied **5-fold cross-validation**, **data augmentation**, and evaluated performance using metrics such as **accuracy, precision, recall, F1-score**, and **confusion matrix** visualizations.

---

### 📁 Dataset
- Source: [Aerial_Landscapes](./dataset_split/train), which can be found on Kaggle:
* [Aerial_Landscapes](https://www.kaggle.com/datasets/balraj98/aerial-landscapes)

- 15 classes, with training and validation folders pre-split.

- Deep learing dataset is loaded using `torchvision.datasets.ImageFolder`.

---
### Libraries

The following Python libraries are used in this notebook:

* pandas
* numpy
* matplotlib
* seaborn
* tensorflow
* keras
* scikit-image (skimage)
* scikit-learn (sklearn)
* opencv-python (cv2)
* tqdm
* glob
* torchvision
* sklearn.metrics
* matplotlib.pyplot
* sklearn.model_selection


### CNNs
## 📁 Dataset Preparation

To create a reduced training dataset (60% of the original images), we provide a script:
```bash
python split_dataset.py
```

## ⚙️ Dependencies
Install using:

```bash
#conda create -n torch310 python=3.10 -y
#conda activate torch310
torchvision, PIL, scikit-learn
```
## Training Scripts:

# 1.Download and Extract: Get archive.zip, extract datasets.

# 2.Process Data: To create a reduced training dataset (60% of the original images)
python split_dataset.py

# 3. Train Models:
/opt/anaconda3/envs/torch310/bin/python train_ResNet18.py

/opt/anaconda3/envs/torch310/bin/python train_EfficientNet-B0.py

# 4.Test Models:
/opt/anaconda3/envs/torch310/bin/python test_ResNet18.py

/opt/anaconda3/envs/torch310/bin/python test_EfficientNet-B0.py

# 5.Evaluation metrics and plots are saved as:

efficientnet_classification_report.txt

efficientnet_confusion_matrix.png

fold_accuracy_plot.png

test_fold_accuracy_plot.png

fold{n}_report.txt, model_fold{n}.pth for each fold

## Example Results:

# ResNet18 5-Fold Validation Accuracy:
Fold 1: 0.9736
Fold 2: 0.9646
Fold 3: 0.9674
Fold 4: 0.9688
Fold 5: 0.9722
Average accuracy: 0.9693

# EfficientNet-B0 5-Fold Validation Accuracy:
Fold 1: 0.9743
Fold 2: 0.9771
Fold 3: 0.9792
Fold 4: 0.9833
Fold 5: 0.9854
Average accuracy: 0.9799

# ResNet18 Confusion Matrix (correct predictions per class):
Agriculture: 471, Airport: 456, Beach: 469, City: 465, Desert: 464, Forest: 473, Grassland: 475, Highway: 456, Lake: 464, Mountain: 456, Parking: 474, Port: 470, Railway: 458, Residential: 471, River: 457.

# EfficientNet-B0 Confusion Matrix (correct predictions per class):
Agriculture: 473, Airport: 472, Beach: 479, City: 475, Desert: 464, Forest: 475, Grassland: 469, Highway: 469, Lake: 467, Mountain: 460, Parking: 471, Port: 474, Railway: 472, Residential: 473, River: 462.


## Folder Structure:
.
├── dataset_split/
│   ├── train/
├── train_ResNet18.py
├── train_EfficientNet-B0.py
├── test_ResNet18.py
├── test_EfficientNet-B0.py
├── split_dataset.py
├── model_fold{n}.pth
├── fold{n}_report.txt
└── README.md

## 📚 Code References

This project leverages the following open-source resources:

- [PyTorch Official Tutorials](https://pytorch.org/tutorials/)
- `torchvision` pretrained models and datasets: https://pytorch.org/vision/stable/models.html (models.efficientnet_b0, models.resnet18）)
- Scikit-learn's classification metrics and cross-validation: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection


### Machine Learning
## Data Processing

The image data is loaded using `keras.utils.image_dataset_from_directory` to create training and validation datasets. The images are resized to 256x256 pixels.

The `extract_images_labels` function is used to extract the images and labels from the TensorFlow datasets into NumPy arrays.

## Feature Extraction

Two feature extraction methods are employed:

* **LBP (Local Binary Pattern):** The `extract_lbp_features` function calculates LBP features for the images.
* **SIFT (Scale-Invariant Feature Transform):** The `extract_sift_features` function calculates SIFT features.

## Model description

The following machine learning models are used for classification:

* **KNN (K-Nearest Neighbors):** Basic and weighted versions.
* **Random Forest:** Basic and weighted versions.
* **SVM (Support Vector Machine):** Basic and weighted versions using SGDClassifier.

Weighted models are implemented to address potential class imbalance in the dataset. Class weights are calculated using `sklearn.utils.class_weight.compute_class_weight`.

## Model Evaluation

The `evaluate_model` function is used to train and evaluate each model. It reports:

* Training time
* Prediction time
* Overall accuracy
* Classification report (precision, recall, F1-score) for each class
* Confusion matrix visualization

## how to run

1.  Ensure you have the required libraries installed.
2.  Download the Aerial Landscapes dataset from the provided Kaggle link and place it in the appropriate directory.
3.  Run the notebook cells sequentially to execute the data loading, feature extraction, model training, and evaluation steps.

## Results

The notebook provides a comprehensive evaluation of different feature extraction and classification model combinations for the landscape classification task. The results, including accuracy and confusion matrices, are presented to compare the performance of each approach.




