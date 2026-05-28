import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import InputLayer
from tensorflow.keras import mixed_precision

# ============================================================
# COMPATIBILITY FIXES
# ============================================================
class CompatInputLayer(InputLayer):
    def __init__(self, **kwargs):
        if 'batch_shape' in kwargs:
            batch_shape = kwargs.pop('batch_shape')
            kwargs['input_shape'] = batch_shape[1:]
        super().__init__(**kwargs)

class DTypePolicy(mixed_precision.Policy):
    def __init__(self, name=None, **kwargs):
        if name is None:
            name = 'float32'
        super().__init__(name)

    @classmethod
    def from_config(cls, config):
        return cls(name=config.get('name', 'float32'))

# ============================================================
# PATHS
# ============================================================
MODEL_PATH   = r"C:\Users\Aarav Gupta\OneDrive\Desktop\segregation and stuff\model_binary.h5"
IMAGE_FOLDER = r"C:\Users\Aarav Gupta\OneDrive\Desktop\input_data"

IMG_SIZE = 224
CLASSES  = ["Recyclable", "Non-Recyclable"]

# ============================================================
# LOAD MODEL
# ============================================================
model = load_model(
    MODEL_PATH,
    custom_objects={
        'InputLayer'      : CompatInputLayer,
        'DTypePolicy'     : DTypePolicy,
        'CompatInputLayer': CompatInputLayer
    },
    compile=False
)

print("✅ Model loaded")

# ============================================================
# CHECK FOLDER
# ============================================================
if not os.path.exists(IMAGE_FOLDER):
    print("❌ Image folder does not exist")
    exit()

image_files = [
    f for f in os.listdir(IMAGE_FOLDER)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

if not image_files:
    print("❌ No images found in folder")
    exit()

print(f"📂 Found {len(image_files)} images")

# ============================================================
# PROCESS IMAGES
# ============================================================
for file_name in image_files:
    image_path = os.path.join(IMAGE_FOLDER, file_name)
    print("\nProcessing:", file_name)

    img = cv2.imdecode(
        np.fromfile(image_path, dtype=np.uint8),
        cv2.IMREAD_COLOR
    )

    if img is None:
        print("❌ Could not read image")
        continue

    # Convert grayscale to BGR if needed
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Preprocess
    img_resized    = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_normalized = img_resized.astype("float32") / 255.0
    img_input      = np.expand_dims(img_normalized, axis=0)

    # ============================================================
    # PREDICTION
    # ============================================================
    preds      = model.predict(img_input, verbose=0)
    class_id   = int(np.argmax(preds))
    confidence = float(np.max(preds))

    label = CLASSES[class_id]

    # 🔥 YOUR CONDITION: force Non-Recyclable if confidence < 20%
    if confidence < 0.2:
        label = "Non-Recyclable"

    # ============================================================
    # OUTPUT
    # ============================================================
    print(f"Result: {label} ({confidence*100:.1f}%)")

    color = (0, 255, 0) if label == "Recyclable" else (0, 0, 255)

    cv2.putText(img, label, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Prediction", img)

    # Press ESC to exit early
    if cv2.waitKey(0) & 0xFF == 27:
        break

cv2.destroyAllWindows()
print("\n✅ Finished processing all images")