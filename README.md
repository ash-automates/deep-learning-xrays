# Chest X-Ray Disease Classification

Full-stack deep learning application for detecting pulmonary diseases from chest X-rays using transfer learning (ResNet50/DenseNet121/EfficientNet) + FastAPI backend + interactive web UI.

**Disease Classes**: Healthy, Viral Pneumonia, Bacterial Pneumonia, COVID-19

## Quick Start (GitHub Codespaces)

```bash
# Install system dependency for OpenCV
sudo apt-get update && sudo apt-get install -y libgl1

# Setup Python environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Train model (synthetic data demo, ~1 min)
python train.py

# Start server
export CLASSES="healthy,viral_pneumonia,bacterial_pneumonia,covid19"
uvicorn server:app --host 0.0.0.0 --port 8000

# Access UI: Ports tab → Open in Browser on port 8000
```

## Key Features

- **CLAHE Enhancement**: Medical-grade contrast enhancement for X-ray detail extraction
- **Class Weighting**: Prevents majority class bias (balanced predictions)
- **X-ray Validation**: Rejects non-medical images (photos, pie charts) automatically
- **Strong Augmentation**: Rotation ±25°, zoom ±30%, brightness ±20% for robust training
- **Full Testing**: pytest suite for preprocessing & API validation

## Project Structure

```
preprocessing.py      # CLAHE, augmentation, normalization
model.py             # Transfer learning CNN
train.py             # Training with class weighting
server.py            # FastAPI + X-ray validation
frontend/index.html  # Drag-drop upload UI
tests/               # pytest suite
```## Training Options

### Option 1: Quick Demo (Synthetic Data)
```bash
python train.py  # Trains in ~1 min, generates resnet50_chest_xray_classifier.h5
```
**Note**: Synthetic data is weak (~25% accuracy). Use real medical datasets for production.

### Option 2: Real Dataset (Production)
```bash
# Organize images: dataset_root/class_name/*.jpg
export DATASET_DIR=/path/to/dataset
export CLASSES="normal,pneumonia,covid19"
export EPOCHS=10
python train.py
```

**Recommended Datasets**:
- [ChestX-ray Pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) (~6k images)
- [COVIDx](https://github.com/lindawangg/COVID-Net) (~16k+ images)
- [NIH Chest X-ray](https://www.nih.gov/news-events/news-releases) (112k+ images)

**Training Outputs** (generated locally, in .gitignore):
- `resnet50_chest_xray_classifier.h5` – Model weights
- `training_history.png`, `confusion_matrix.png`, `roc_curves.png` – Visualizations

---

## Running the Web App

```bash
source venv/bin/activate
export CLASSES="healthy,viral_pneumonia,bacterial_pneumonia,covid19"
uvicorn server:app --host 0.0.0.0 --port 8000
```

**Access UI**:
- **Codespaces**: Ports tab → Right-click 8000 → "Open in Browser"
- **Local**: http://localhost:8000/frontend

**Usage**: Drag-drop chest X-ray image → Click "Run inference" → View predicted disease + confidence

---

## Testing

```bash
pytest -v  # Runs 4 tests (preprocessing + API validation)
```

---

## Environment Variables

| Variable | Default | Example |
|----------|---------|---------|
| `DATASET_DIR` | None (uses synthetic) | `/home/user/xray_data` |
| `CLASSES` | healthy,viral_pneumonia,bacterial_pneumonia,covid19 | `normal,pneumonia` |
| `EPOCHS` | 3 | `10` |
| `BATCH_SIZE` | 32 | `16` |
| `MODEL_NAME` | resnet50 | `densenet121`, `efficientnetb0` |
| `MODEL_PATH` | resnet50_chest_xray_classifier.h5 | `/path/to/model.h5` |

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `libGL.so.1: cannot open` | `sudo apt-get install -y libgl1` |
| "Not an X-ray" error | Model rejects colored/non-medical images by design |
| Port 8000 in use | `uvicorn server:app --port 8001` |
| Module not found | Activate venv: `source venv/bin/activate` |
| Codespaces port forwarding | Click **Ports** tab → **Open in Browser** |

---

## Key Technical Concepts

**Transfer Learning**: Pre-trained ImageNet models (ResNet/DenseNet/EfficientNet) provide learned features → train only classification head → faster convergence + better accuracy with limited data.

**Class Weighting**: Without it, model predicts majority class 80% of the time. With `compute_class_weight('balanced')`, each disease gets equal importance → clinically useful predictions.

**CLAHE**: Medical imaging standard for contrast enhancement. Adaptive histogram equalization per 8×8 tile preserves X-ray detail without oversaturation.

**X-ray Validation**: Colorfulness < 18 (grayscale check) + edge density > 0.5% (anatomical features) + aspect ratio 0.4-2.5 → rejects photos/pie charts automatically.

---

## Learning Outcomes

✅ Medical image classification with CNNs  
✅ Transfer learning & fine-tuning  
✅ Class imbalance handling (weighted loss)  
✅ CLAHE preprocessing for medical images  
✅ FastAPI REST API development  
✅ Frontend-backend integration  
✅ pytest testing & validation  

---

## License

MIT License
