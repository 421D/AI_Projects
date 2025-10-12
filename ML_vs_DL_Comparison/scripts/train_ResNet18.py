import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

data_dir = 'dataset_split/train'
batch_size = 32
num_epochs = 10
learning_rate = 1e-4
num_classes = 15
k_folds = 5

# Data enhancement and pre-processing
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
kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
fold_results = []

# Cross-validation main loop
for fold, (train_idx, val_idx) in enumerate(kfold.split(full_dataset)):
    print(f"\n================ Fold {fold + 1}/{k_folds} ================")

    full_dataset.transform = train_transform
    train_subset = Subset(full_dataset, train_idx)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)

    full_dataset.transform = val_transform
    val_subset = Subset(full_dataset, val_idx)
    val_loader = DataLoader(val_subset, batch_size=batch_size)

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        print(f"\n🔁 Epoch {epoch+1}/{num_epochs}")
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
            torch.save(model.state_dict(), f'model_fold{fold+1}.pth')
            print(f"✅ Best model saved for Fold {fold+1}")

            # save classification report
            report = classification_report(
                all_labels, all_preds,
                target_names=full_dataset.classes,
                digits=4
            )
            with open(f"fold{fold+1}_report.txt", "w") as f:
                f.write(report)
            print("📄 Classification report saved.")

    print(f"📌 Fold {fold+1} best validation accuracy: {best_val_acc:.4f}")
    fold_results.append(best_val_acc)

print("\n🎉 Cross-validation complete.")
for i, acc in enumerate(fold_results):
    print(f"Fold {i+1}: Best Val Acc = {acc:.4f}")
print(f"Average Val Acc: {np.mean(fold_results):.4f}")

plt.figure(figsize=(8, 5))
plt.plot(range(1, k_folds + 1), fold_results, marker='o', linestyle='-', linewidth=2, color='orange')
plt.title("5-Fold Cross-Validation Accuracy")
plt.xlabel("Fold")
plt.ylabel("Validation Accuracy")
plt.ylim(0.90, 1.00)
plt.grid(True)
plt.xticks(range(1, k_folds + 1))

avg = np.mean(fold_results)
plt.axhline(avg, color='gray', linestyle='--', label=f"Avg = {avg:.4f}")
plt.legend()
plt.tight_layout()
plt.savefig("fold_accuracy_plot.png")
print("📈 Saved: fold_accuracy_plot.png")
