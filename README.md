# Image Classification with PyTorch: CNN & ResNet on CIFAR-10

## 1. Project Title and Overview

This repository contains a **deep learning project** for **image classification** using **PyTorch**, implementing and comparing two approaches:

- **Custom CNN** — A Convolutional Neural Network built from scratch.
- **ResNet (Transfer Learning)** — A pretrained ResNet18 model fine-tuned on CIFAR-10.

Both models are **trained and evaluated on the CIFAR-10 dataset**, with data augmentation, a full training pipeline, and evaluation metrics. The project is suitable for **portfolio presentation** and for **study/revision** of CNN and image classification concepts.

---

## 2. Objectives of the Project

- Implement a **CNN from scratch** and understand its layer-by-layer design.
- Use **ResNet** (pretrained) with **transfer learning** and optional fine-tuning.
- Train and evaluate on **CIFAR-10** with reproducible pipelines.
- Apply **data augmentation** to improve generalization.
- Compare **training pipeline**, **evaluation metrics**, and **accuracy** across models.
- Achieve **~90%+ test/validation accuracy** on CIFAR-10.

---

## 3. Dataset Description (CIFAR-10)

**CIFAR-10** is a standard benchmark dataset for image classification.

| Property        | Value                          |
|----------------|---------------------------------|
| **Images**     | 60,000 colour images            |
| **Resolution** | 32×32 pixels                    |
| **Channels**   | 3 (RGB)                         |
| **Train set**  | 50,000 images (5,000 per class) |
| **Test set**   | 10,000 images (1,000 per class) |
| **Classes**    | 10                              |

**Class labels:** airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.

The dataset is **balanced** (equal samples per class), which simplifies training and metric interpretation. It is available via `torchvision.datasets.CIFAR10` and downloads automatically when `download=True`.

---

## 4. Model Architectures

### 4.1 CNN (From Scratch)

The custom CNN follows a classic pattern: **convolutional feature extraction** followed by **fully connected classification**.

**Design principles:**

- **Convolutional layers** learn spatial, hierarchical features (edges → textures → object parts).
- **Batch Normalization** stabilizes training and speeds convergence.
- **ReLU** provides non-linearity.
- **Max Pooling** reduces spatial dimensions and adds slight invariance to small shifts.
- **Dropout** in the classifier reduces overfitting.

**Architecture summary:**

| Stage   | Layers / operations                         | Output shape (approx.) |
|--------|----------------------------------------------|------------------------|
| Input  | —                                            | (B, 3, 32, 32)         |
| Conv1  | Conv2d(3→64, 3×3, pad=1), BN, ReLU          | (B, 64, 32, 32)        |
| Conv2  | Conv2d(64→64, 3×3, pad=1), BN, ReLU, MaxPool | (B, 64, 16, 16)        |
| Conv3  | Conv2d(64→128, 3×3, pad=1), BN, ReLU         | (B, 128, 16, 16)       |
| Conv4  | Conv2d(128→128, 3×3, pad=1), BN, ReLU, MaxPool | (B, 128, 8, 8)      |
| Conv5  | Conv2d(128→256, 3×3, pad=1), BN, ReLU, MaxPool | (B, 256, 4, 4)      |
| Classifier | Flatten, Linear(256×4×4 → 512), ReLU, Dropout(0.5), Linear(512 → 10) | (B, 10) |

**Concept recap:**

- **Convolution:** sliding filters over the image to detect local patterns.
- **Padding:** keeps spatial size under control (e.g. `padding=1` with 3×3 kernels).
- **Pooling:** downsampling (e.g. 2×2 MaxPool halves width and height).

---

### 4.2 ResNet and Residual Connections

**ResNet** (Residual Network) addresses the **vanishing gradient** problem in very deep networks by introducing **skip connections** (residual connections).

**Residual block idea:**

Instead of learning \( H(x) \) directly, the block learns the **residual** \( F(x) = H(x) - x \). The output is:

\[
y = F(x) + x
\]

- **Benefits:** gradients can flow directly through the identity path, so deeper networks remain trainable.
- **Identity shortcut:** \( x \) is added to the output of the convolutional branch, enabling stable training for many layers.

**In this project:**

- **ResNet18** from `torchvision.models` is used (18 layers, pretrained on ImageNet).
- **Transfer learning:**  
  - Pretrained weights are used for feature extraction.  
  - The final fully connected layer is **replaced** with `nn.Linear(512, 10)` for the 10 CIFAR-10 classes.  
  - Most layers are **frozen**; only **layer4** (last residual block) and the new **fc** are trained.
- Input size is **resized to 224×224** to match ImageNet, and **ImageNet normalization** (mean/std) is applied.

**Concept recap:**

- **Transfer learning:** reuse features learned on a large dataset (e.g. ImageNet) for a smaller task (CIFAR-10).
- **Fine-tuning:** unfreezing some layers (here, layer4 + fc) and training with a small learning rate.

---

## 5. Data Augmentation Techniques Used

Data augmentation artificially expands the training set and improves generalization.

### CNN notebook

| Technique              | Parameters              | Purpose                          |
|------------------------|-------------------------|----------------------------------|
| **RandomCrop**         | 32, padding=4            | Random 32×32 crop; adds scale/position variance |
| **RandomHorizontalFlip** | —                      | Mirror images; preserves semantics for CIFAR-10 |
| **ColorJitter**        | brightness=0.2, contrast=0.2 | Varies lighting/contrast        |
| **Normalize**           | mean=0.5, std=0.5 (per channel) | Zero-mean, unit variance      |

### ResNet notebook

| Technique              | Parameters              | Purpose                          |
|------------------------|-------------------------|----------------------------------|
| **Resize**             | (224, 224)               | Match ResNet/ImageNet input size |
| **RandomHorizontalFlip** | —                      | Same as above                    |
| **Normalize**          | ImageNet mean/std        | Match pretrained statistics      |

**Test/validation:** No randomness (e.g. no flip, no random crop); only resize (for ResNet) and the same normalization as training.

---

## 6. Training Pipeline Explanation

### 6.1 High-level steps

1. **Data loading:** CIFAR-10 with train/test transforms and `DataLoader` (shuffle=True for training).
2. **Model:** Move model to GPU if available (`device`).
3. **Loss:** `nn.CrossEntropyLoss()` (classification).
4. **Optimizer:** Adam.
5. **Loop:** For each epoch:
   - **Train:** `model.train()`, forward pass, loss backward, optimizer step; optional scheduler step.
   - **Evaluate:** `model.eval()`, no gradients, run on test/validation set.
6. **Checkpointing:** Save best model (e.g. by validation accuracy).

### 6.2 CNN training setup

- **Epochs:** 30  
- **Optimizer:** Adam, lr = 0.001  
- **Scheduler:** `StepLR`, step_size=10, gamma=0.5 (halve LR every 10 epochs)  
- **Batch size:** 64 (in the provided loader)  
- **Checkpoint:** Best model by validation accuracy → `best_model.pth`

### 6.3 ResNet training setup

- **Epochs:** 10 (fewer due to transfer learning)  
- **Optimizer:** Adam, lr = 1e-4 (only for unfrozen parameters)  
- **Frozen:** All parameters except `layer4` and `fc`  
- **Batch size:** 64  

---

## 7. Evaluation Metrics

- **Accuracy:** Primary metric — proportion of correct predictions on the test/validation set.  
  \[
  \text{Accuracy} = \frac{\text{Number of correct predictions}}{\text{Total number of samples}}
  \]
- **Loss:** Cross-entropy loss (training and, if computed, validation) to monitor convergence.
- **Per-epoch tracking:** Training loss and validation/test accuracy printed each epoch.
- **Best model:** Selected by highest validation accuracy (CNN) or final test accuracy (ResNet); suitable for reporting a single “best” result.

For a more detailed analysis, you can extend the pipeline with **per-class accuracy**, **confusion matrix**, or **precision/recall/F1**.

---

## 8. Results and Accuracy Comparison

| Model        | Description                    | Best test/validation accuracy |
|-------------|--------------------------------|--------------------------------|
| **Custom CNN** | From scratch, 30 epochs       | **89.11%**                     |
| **ResNet18**   | Pretrained, fine-tuned, 10 epochs | **91.95%**                  |

**Observations:**

- ResNet achieves **higher accuracy** with **fewer epochs** thanks to pretrained features and residual architecture.
- The custom CNN reaches **~89%** without pretraining, showing that a well-designed pipeline (augmentation, BatchNorm, scheduler) is sufficient for strong CIFAR-10 performance.
- Both models meet the **~90%+** goal (ResNet clearly; CNN close and improvable with more tuning).

---

## 9. Key Learnings and Concepts Covered

- **CNNs:** Convolutions, padding, stride, pooling, feature maps, and stacking layers for hierarchy.
- **Batch Normalization:** Normalizing activations to stabilize and accelerate training.
- **Residual connections:** Skip connections, residual learning, and training very deep networks.
- **Transfer learning:** Using pretrained models, freezing vs. fine-tuning, and adapting the classifier head.
- **Data augmentation:** Random crop, flip, color jitter, and normalization for better generalization.
- **Training loop:** Forward pass, loss, backward pass, optimizer step, train vs. eval mode.
- **Learning rate scheduling:** StepLR (and the idea of reducing LR over time).
- **Evaluation:** Accuracy, checkpointing the best model, and reporting test/validation results.

---

## 10. How to Run the Project

### Prerequisites

- **Python:** 3.7+  
- **PyTorch** and **torchvision** (with CUDA if you want GPU)

### Step-by-step setup

**1. Clone or download the project** (e.g. ensure `cnn_pytorch.ipynb` and `resnet_pytorch.ipynb` are in the same folder as this README).

**2. Create and activate a virtual environment (recommended):**

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate
```

**3. Install PyTorch and torchvision:**

```bash
# CPU only (example)
pip install torch torchvision

# GPU (CUDA) — adjust cuda version as needed
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**4. Install other dependencies (if used in the notebooks):**

```bash
pip install numpy pandas matplotlib
```

**5. Run the notebooks:**

- **CNN:** Open and run all cells in `cnn_pytorch.ipynb`.  
  - CIFAR-10 will download to `./data` on first run.  
  - Best model is saved as `best_model.pth` in the notebook’s working directory.

- **ResNet:** Open and run all cells in `resnet_pytorch.ipynb`.  
  - Same data directory; images are resized to 224×224 inside the transform.

**6. (Optional) Use a different data root:**  
Change `root='./data'` in the dataset constructors if you want data stored elsewhere.

---

## 11. Future Improvements

- **CNN:** Try deeper/wider CNNs, more augmentation (e.g. CutOut, RandomRotation), and learning rate warm-up or OneCycleLR.
- **ResNet:** Experiment with unfreezing more layers (e.g. layer3 + layer4), different learning rates for backbone vs. classifier, or ResNet34/50.
- **Evaluation:** Add confusion matrix, per-class accuracy, precision/recall/F1, and optional ROC curves.
- **Reproducibility:** Set `torch.manual_seed`, `np.random.seed`, and `torch.backends.cudnn.deterministic` where appropriate.
- **Logging:** Use TensorBoard or Weights & Biases for loss/accuracy curves and hyperparameter tracking.
- **Inference script:** Add a small script to load a saved model and run inference on single images or a test set.

---

## 12. Folder Structure Explanation

Suggested layout for the project:

```
project_root/
├── README.md                 # This file
├── cnn_pytorch.ipynb         # Custom CNN training and evaluation
├── resnet_pytorch.ipynb      # ResNet18 transfer learning and evaluation
└── requirements.txt         # Optional: pip freeze > requirements.txt
```

- **README.md:** Project overview, setup, and revision-oriented explanations.  
- **Notebooks:** Self-contained pipelines (data, model, train, evaluate).  


---

## License and Citation

If you use CIFAR-10, consider citing the dataset. PyTorch and torchvision are under their respective licenses. This README and the described code are intended for learning and portfolio use.

---

*README designed for GitHub presentation and for study/revision of CNN and image classification with PyTorch.*
