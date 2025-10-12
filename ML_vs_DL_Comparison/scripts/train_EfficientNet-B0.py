import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

data_dir = 'dataset_split/train'
batch_size = 32
num_epochs = 10
learning_rate = 1e-4
k_folds = 5

# Data Enhancement and Preprocessing
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ToTensor()
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

full_dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)
num_classes = len(full_dataset.classes)
kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
fold_results = []

# Cross Validation Master Loop
for fold, (train_idx, val_idx) in enumerate(kfold.split(full_dataset)):
    print(f"\n================ Fold {fold + 1}/{k_folds} ================")

    full_dataset.transform = train_transform
    train_subset = Subset(full_dataset, train_idx)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)

    full_dataset.transform = val_transform
    val_subset = Subset(full_dataset, val_idx)
    val_loader = DataLoader(val_subset, batch_size=batch_size)

    # Model Initialization
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_val_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        print(f"Epoch {epoch+1} | Loss: {total_loss:.4f} | Train Acc: {train_acc:.4f}")

        model.eval()
        correct = 0
        total = 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_acc = correct / total
        print(f"→ Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f'effnet_fold{fold+1}.pth')
            print(f"✅ Best model saved for Fold {fold+1}")

            # 保存 classification report
            report = classification_report(
                all_labels, all_preds,
                target_names=full_dataset.classes,
                digits=4
            )
            with open(f'effnet_fold{fold+1}_report.txt', 'w') as f:
                f.write(report)
            print("📄 Classification report saved.")

    fold_results.append(best_val_acc)

plt.figure(figsize=(8, 5))
plt.plot(range(1, k_folds + 1), fold_results, marker='o', linestyle='-', color='green', label="EfficientNet-B0 Val Acc")
plt.title("EfficientNet-B0 5-Fold Validation Accuracy")
plt.xlabel("Fold")
plt.ylabel("Validation Accuracy")
plt.grid(True)

avg = np.mean(fold_results)
plt.axhline(avg, color='gray', linestyle='--', label=f"Avg = {avg:.4f}")
plt.legend()
plt.tight_layout()
plt.savefig("effnet_val_accuracy.png")
print("📈 Saved: effnet_val_accuracy.png")

print("🎉 Training & Cross-validation complete.")
