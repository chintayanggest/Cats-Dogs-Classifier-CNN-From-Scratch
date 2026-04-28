# 🐱🐶 Cat vs. Dog Image Classifier — CNN From Scratch

A complete end-to-end Deep Learning project where a Convolutional Neural Network (CNN) is designed, debugged, and trained **from scratch** using TensorFlow/Keras — no pre-trained models, no shortcuts.

> **Final Result: 84.5% Validation Accuracy** after 3 architecture iterations and real-world debugging.

---

## 📌 Project Highlights

- Built a CNN from scratch without transfer learning (no ResNet, no VGG)
- Diagnosed and fixed real Deep Learning bugs: **Overfitting**, **Dying ReLUs**, and **Exploding Gradients**
- Tracked the full engineering journey across 3 model versions
- Trained locally with GPU acceleration

---

## 🚀 The Engineering Journey

The model wasn't built in one shot — it was **iterated and debugged** through 3 versions, each solving a real problem.

### V1 — Baseline (The Overfitting Disaster)
A simple 2-layer CNN that memorized the training data rather than learning.

| Metric | Score |
|---|---|
| Training Accuracy | 97% |
| Validation Accuracy | 55% |

**Diagnosis:** Severe overfitting. The model memorized pixel locations instead of learning spatial features like ears and whiskers.

---

### V2 — Combatting Overfitting
Introduced `Dropout(0.5)` and **Data Augmentation** (Random Flip, Rotation, Zoom) to force generalization.

| Metric | Score |
|---|---|
| Validation Accuracy | 80.4% |

**Diagnosis:** Overfitting eliminated, but accuracy plateaued. The 2-layer architecture had hit its **model capacity limit** — it wasn't deep enough to understand complex textures like fur.

---

### V3 — Deeper Network + A Fatal Bug 🐛

Added a 3rd convolutional block (128 filters) for higher-level feature extraction. But the model crashed immediately:

- Loss flatlined at exactly **`0.693`**
- Accuracy froze at **`50.0%`** (pure random guessing)

**Root Cause:** The mathematical signature `-ln(0.5) = 0.693` confirmed the model was completely blind. Raw pixel values (0–255) fed into a deeper network caused **exploding gradients**, which killed all neurons (Dying ReLUs).

**Fix:** Added a `layers.Rescaling(1./255)` preprocessing layer to normalize inputs to the [0, 1] range.

---

### 🏆 V3 Final Results

| Metric | Score |
|---|---|
| Validation Accuracy | **85.5%** |
| Validation Loss | **0.365** |
| Training Epochs | 45 (with Early Stopping) |

---

## 🧠 Final Model Architecture

The full architecture is documented inside the notebook. Here's a high-level overview:

```
Input (150×150×3)
│
├── Data Augmentation  →  RandomFlip, RandomRotation, RandomZoom
├── Preprocessing      →  Rescaling(1./255)  ← Critical bug fix
│
├── Conv Block 1       →  Conv2D(32) + MaxPooling
├── Conv Block 2       →  Conv2D(64) + MaxPooling
├── Conv Block 3       →  Conv2D(128) + MaxPooling
│
├── Flatten
├── Dropout(0.5)       ← Overfitting prevention
├── Dense(128, relu)
└── Dense(1, sigmoid)  ← Binary output: Cat or Dog
```

---

## 💾 Pre-Trained Model

The trained `.keras` model file is hosted on Hugging Face (too large for GitHub).

🔗 **[Download Model on Hugging Face](https://huggingface.co/ChintaYanggest/cat-dog-classifier)**

### Run Inference on Your Own Image

```python
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

model = load_model('cat_vs_dog_model.keras')

img = image.load_img('your_image.jpg', target_size=(150, 150))
img_array = np.expand_dims(image.img_to_array(img), axis=0)

prediction = model.predict(img_array)

print("🐱 Cat!" if prediction[0][0] < 0.5 else "🐶 Dog!")
```

> **Note:** No manual rescaling needed — the model handles `1./255` normalization internally.

---

## 📓 Full Walkthrough

The complete code, training logs, and visualizations are inside the Jupyter Notebook:

📄 **`Cat_vs_Dog_CNN.ipynb`**

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.x | Core language |
| TensorFlow / Keras | Model building & training |
| NumPy | Array operations |
| Kaggle Dataset | 25,000 cat & dog images |
| GPU (Local) | Hardware-accelerated training |
