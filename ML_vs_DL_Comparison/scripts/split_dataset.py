import os
import shutil
from sklearn.model_selection import train_test_split
from glob import glob

# Original dataset paths
src_root = "Aerial_Landscapes"

# Outputs path
dst_root = "dataset_split"
train_dir = os.path.join(dst_root, "train")

# Creating a directory of training data
os.makedirs(train_dir, exist_ok=True)

# Per-category processing
for class_name in os.listdir(src_root):
    src_class_dir = os.path.join(src_root, class_name)
    if not os.path.isdir(src_class_dir):
        continue

    images = glob(os.path.join(src_class_dir, "*.jpg"))
    if len(images) == 0:
        continue

    # Randomly select 60% of the images
    train_imgs, _ = train_test_split(images, train_size=0.6, random_state=42)

    # Creating the Create target category folder
    dst_class_dir = os.path.join(train_dir, class_name)
    os.makedirs(dst_class_dir, exist_ok=True)

    for img_path in train_imgs:
        shutil.copy(img_path, dst_class_dir)

    print(f"✅ Processed class: {class_name} | Selected {len(train_imgs)} images")

print("\n🎉 Done: 60% training dataset created under dataset_split/train/")
