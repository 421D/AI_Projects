import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

device = torch.device("mps" if torch.cuda.is_available() else "cpu")

# Parameters
model_prefix = 'model_fold' 
num_classes = 15
batch_size = 32
k_folds = 5

data_dir = 'dataset_split/train'  # In cross-validation we are validating the different divisions of train

# Image preprocessing (for validation)
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Dataset loading (full dataset + folding)
full_dataset = datasets.ImageFolder(root=data_dir, transform=val_transform)
kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)

all_true = []
all_pred = []
fold_accuracies = []

for fold, (_, val_idx) in enumerate(kfold.split(full_dataset)):
    print(f"\n===== Evaluating Fold {fold + 1}/{k_folds} =====")

    val_subset = Subset(full_dataset, val_idx)
    val_loader = DataLoader(val_subset, batch_size=batch_size)

    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(f"{model_prefix}{fold+1}.pth", map_location=device))
    model = model.to(device)
    model.eval()

    fold_preds = []
    fold_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            fold_preds.extend(preds.cpu().numpy())
            fold_labels.extend(labels.cpu().numpy())

    all_pred.extend(fold_preds)
    all_true.extend(fold_labels)

    # Accuracy per fold
    correct = np.sum(np.array(fold_preds) == np.array(fold_labels))
    accuracy = correct / len(fold_labels)
    fold_accuracies.append(accuracy)
    print(f"Fold {fold + 1} Accuracy: {accuracy:.4f}")

print("\n✅ Overall Classification Report:")
print(classification_report(all_true, all_pred, target_names=full_dataset.classes))

print("Confusion Matrix:")
print(confusion_matrix(all_true, all_pred))
cm = confusion_matrix(all_true, all_pred)
labels = full_dataset.classes

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("ResNet18 Confusion Matrix")
plt.tight_layout()
plt.savefig("resnet18_confusion_matrix.png")
print("🖼️ Saved: resnet18_confusion_matrix.png")

plt.figure(figsize=(8, 5))
plt.plot(range(1, k_folds + 1), fold_accuracies, marker='o', linestyle='-', color='blue')
plt.title("5-Fold Validation Accuracy")
plt.xlabel("Fold")
plt.ylabel("Accuracy")
plt.ylim(0.0, 1.0)
plt.grid(True)
avg_acc = np.mean(fold_accuracies)
plt.axhline(avg_acc, color='gray', linestyle='--', label=f"Avg = {avg_acc:.4f}")
plt.legend()
plt.tight_layout()
plt.savefig("test_fold_accuracy_plot.png")
print("\n📈 Saved: test_fold_accuracy_plot.png")
