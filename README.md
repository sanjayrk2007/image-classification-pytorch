# 🧠 Image Classification: CNN & ResNet on CIFAR-10

> **PyTorch** · Custom CNN from scratch · ResNet18 transfer learning · **~90%+ accuracy**

Two approaches to the same goal: train and compare a **hand-built CNN** and a **pretrained ResNet** on CIFAR-10. Built for portfolio showcase and quick revision of CNN & image classification.

---

## ▶ What's in this repo

| | **Custom CNN** | **ResNet18** |
|--|----------------|--------------|
| **Approach** | Built from scratch | Pretrained, fine-tuned |
| **Accuracy** | **88.44%** | **91.95%** |
| **Epochs** | 30 | 10 |
| **Notebook** | `cnn_pytorch.ipynb` | `resnet_pytorch.ipynb` |

---

## 📦 Dataset: CIFAR-10

60K colour images (32×32), 10 classes — *airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck*.  
50K train / 10K test, balanced. Auto-downloads via `torchvision`.

---

## 🏗 Model architectures (at a glance)

**CNN** — Conv blocks (3→64→128→256) with BatchNorm, ReLU, MaxPool → Flatten → FC(512) + Dropout(0.5) → 10 classes. Classic *conv feature extraction + classifier*.

**ResNet** — Skip connections: output = `F(x) + x` so gradients flow through identity; enables deep nets. We use **ResNet18** (ImageNet-pretrained), replace the final FC with 10 classes, freeze all except **layer4** and **fc**, then fine-tune. Inputs resized to 224×224 + ImageNet normalization.

---

## 🔄 Data augmentation

**CNN:** `RandomCrop(32, padding=4)` · `RandomHorizontalFlip()` · `ColorJitter(brightness=0.2, contrast=0.2)` · Normalize(0.5, 0.5)  
**ResNet:** `Resize(224,224)` · `RandomHorizontalFlip()` · ImageNet normalize  

Test/val: no randomness — only resize (ResNet) and same normalization.

---

## 📈 Training & evaluation

**Pipeline:** Load CIFAR-10 → model to device → CrossEntropyLoss + Adam → train loop (forward, backward, step) → eval on test/val → save best checkpoint.

| | CNN | ResNet |
|--|-----|--------|
| Optimizer | Adam, lr=0.001 | Adam, lr=1e-4 (unfrozen only) |
| Scheduler | StepLR (×0.5 every 10 epochs) | — |
| Metric | Validation accuracy | Test accuracy |
| Checkpoint | `best_model.pth` | — |

---

## 🚀 Run it

**Requirements:** Python 3.7+, PyTorch, torchvision. (GPU optional but faster.)

```bash
pip install -r requirements.txt
```

Then open and run **all cells** in `cnn_pytorch.ipynb` and/or `resnet_pytorch.ipynb`. Data goes to `./data` on first run.

---

## 📁 Project layout

```
├── README.md
├── requirements.txt
├── cnn_pytorch.ipynb      # CNN from scratch
├── resnet_pytorch.ipynb   # ResNet18 transfer learning
├── data/                  # CIFAR-10 (auto-created)
└── best_model.pth         # Best CNN weights (after training)
```

---

## 💡 Concepts covered

CNNs (conv, padding, pooling) · BatchNorm · Residual connections · Transfer learning & fine-tuning · Data augmentation · Train/eval loop · LR scheduling · Checkpointing

---

## 🔮 Possible next steps

Deeper/wider CNN · CutOut / RandomRotation · Unfreeze more ResNet layers · Confusion matrix & per-class metrics · TensorBoard · Reproducibility seeds · Standalone inference script

---

*CIFAR-10 citation recommended when using the dataset. PyTorch/torchvision under their respective licenses.*
