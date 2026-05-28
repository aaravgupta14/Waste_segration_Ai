import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, RandomFlip, RandomRotation, RandomZoom
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt

# ======================
# CONFIG
# ======================
DATASET_PATH = r"C:\Users\Aarav Gupta\OneDrive\Desktop\waste_dataset"
SAVE_PATH    = r"C:\Users\Aarav Gupta\OneDrive\Desktop\segregation and stuff\model_binary.h5"
IMG_SIZE     = 224
BATCH_SIZE   = 32
EPOCHS       = 20

RECYCLABLE   = ["cardboard", "glass", "metal", "paper", "plastic"]

# ======================
# LOAD DATASET
# ======================
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    f"{DATASET_PATH}/train",
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    f"{DATASET_PATH}/val",
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
print("Classes found:", class_names)

recyclable_indices = [class_names.index(c) for c in RECYCLABLE if c in class_names]
print("Recyclable indices:", recyclable_indices)

# ======================
# NORMALIZE
# ======================
train_ds = train_ds.map(lambda x, y: (x / 255.0, y))
val_ds   = val_ds.map(lambda x, y: (x / 255.0, y))

# ======================
# BINARY LABEL MAPPING
# ======================
def to_binary(images, labels):
    binary_labels = tf.where(
        tf.reduce_any(
            tf.equal(tf.expand_dims(labels, -1), recyclable_indices),
            axis=1
        ),
        0,  # Recyclable
        1   # Non-Recyclable
    )
    return images, binary_labels

train_ds = train_ds.map(to_binary)
val_ds   = val_ds.map(to_binary)

# ======================
# DATA AUGMENTATION
# ======================
data_augmentation = tf.keras.Sequential([
    RandomFlip("horizontal"),
    RandomRotation(0.2),
    RandomZoom(0.2),
])

train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds   = val_ds.prefetch(tf.data.AUTOTUNE)

# ======================
# CLASS WEIGHTS
# ======================
num_recyclable     = 1500  # 250 x 5 classes
num_non_recyclable = 250   # trash only
total              = num_recyclable + num_non_recyclable

class_weights = {
    0: total / (2 * num_recyclable),      # Recyclable
    1: total / (2 * num_non_recyclable)   # Non-Recyclable
}
print("Class weights:", class_weights)

# ======================
# BUILD MODEL
# ======================
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', padding='same',
           input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.summary()

# ======================
# COMPILE
# ======================
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ======================
# CALLBACKS
# ======================
if os.path.exists(SAVE_PATH):
    os.remove(SAVE_PATH)
    print("🗑️ Old model deleted")

callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        SAVE_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
]

# ======================
# TRAIN
# ======================
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weights,   # ✅ fixes class imbalance
    callbacks=callbacks
)

# ======================
# PLOTS
# ======================
plt.figure()
plt.plot(history.history['accuracy'],     label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Model Accuracy')
plt.legend()
plt.grid()
plt.savefig("binary_accuracy.png")
plt.show()

plt.figure()
plt.plot(history.history['loss'],     label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Model Loss')
plt.legend()
plt.grid()
plt.savefig("binary_loss.png")
plt.show()

# ======================
# FINAL METRICS
# ======================
print("\n" + "="*50)
print("         TRAINING COMPLETE")
print("="*50)
print(f"✅ Final Training Accuracy   : {history.history['accuracy'][-1]*100:.2f}%")
print(f"✅ Final Validation Accuracy : {history.history['val_accuracy'][-1]*100:.2f}%")
print(f"✅ Final Training Loss       : {history.history['loss'][-1]:.4f}")
print(f"✅ Final Validation Loss     : {history.history['val_loss'][-1]:.4f}")
print(f"✅ Model saved to            : {SAVE_PATH}")