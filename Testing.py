import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import numpy as np

# ============================================================
# PATH
# ============================================================
TRAIN_PATH = r"C:\Users\Aarav Gupta\OneDrive\Desktop\waste_dataset\train"

# ============================================================
# LOAD DATASET
# ============================================================
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TRAIN_PATH,
    image_size=(224, 224),
    batch_size=32,
    shuffle=False
)

file_paths = train_ds.file_paths  # ✅ get file names
class_names = train_ds.class_names

# ============================================================
# DEFINE RECYCLABLE CLASSES
# ============================================================
RECYCLABLE = ["cardboard", "glass", "metal", "paper", "plastic"]
recyclable_indices = [class_names.index(c) for c in RECYCLABLE if c in class_names]

# convert to tensor (important)
recyclable_indices = tf.constant(recyclable_indices)

# ============================================================
# CONVERT TO BINARY
# ============================================================
def to_binary(images, labels):
    binary_labels = tf.where(
        tf.reduce_any(
            tf.equal(tf.expand_dims(labels, -1), recyclable_indices),
            axis=1
        ),
        0,
        1
    )
    return images / 255.0, binary_labels

train_ds = train_ds.map(to_binary)

# ============================================================
# LOAD MODEL
# ============================================================
model = tf.keras.models.load_model("model_binary.h5")

# ============================================================
# FIND NON-RECYCLABLE PREDICTIONS
# ============================================================
idx = 0

print("\n🔍 Images predicted as NON-RECYCLABLE:\n")

for images, labels in train_ds:
    preds = model.predict(images, verbose=0).flatten()
    pred_classes = (preds > 0.5).astype(int)

    for i in range(len(images)):
        if pred_classes[i] == 1:
            print(file_paths[idx])   # ✅ file name
        
        idx += 1

print("\n✅ Done")