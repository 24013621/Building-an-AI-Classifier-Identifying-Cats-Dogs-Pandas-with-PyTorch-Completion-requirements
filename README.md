# Developing a Convolutional Neural Network Classification Model Using Transfer Learning

## AIM

To develop an image classification model using transfer learning in PyTorch that classifies images into **cat**, **dog**, or **panda**, using a pre-trained CNN backbone.

## THEORY

The Cats vs Dogs vs Pandas dataset consists of labeled images belonging to three classes. Training a convolutional neural network from scratch on a relatively small image dataset is prone to overfitting and requires significant compute. Transfer learning addresses this by reusing a CNN (ResNet18) already pre-trained on ImageNet: the convolutional feature-extraction layers, which have already learned general-purpose visual features (edges, textures, shapes), are frozen, and only a new, lightweight classifier head is trained on top of them for the target 3-class problem. This drastically reduces the number of trainable parameters and the amount of data/epochs needed to reach good accuracy.

## Neural Network Model

```
Input Image (224x224x3)
        │
        ▼
[ResNet18 Convolutional Backbone] (pretrained on ImageNet, FROZEN)
        │
        ▼
   Global Average Pool
        │
        ▼
   Linear(in_features -> 256)
        │
        ▼
        ReLU
        │
        ▼
   Dropout(p=0.5)
        │
        ▼
   Linear(256 -> 3)
        │
        ▼
Output: cat / dog / panda
```

## DESIGN STEPS

### STEP 1: Environment Setup
Verify Python 3.9+, PyTorch/torchvision installation, and confirm GPU/CUDA availability before training.

### STEP 2: Data Collection and Understanding
Download the Cats vs Dogs vs Pandas dataset from Kaggle and inspect the class folders and image counts.

### STEP 3: Data Preparation
Organize images into `train/` and `test/` folders per class, load them with `torchvision.datasets.ImageFolder`, resize to 224x224, normalize with ImageNet mean/std, and apply data augmentation (flip, rotation, crop) to the training set.

### STEP 4: Model Architecture Design
Load a pretrained ResNet18, freeze its convolutional layers, and replace the final fully-connected layer with a custom head: `Linear -> ReLU -> Dropout(0.5) -> Linear(3)`.

### STEP 5: Model Training and Optimization
Train only the new classifier head for 10-15 epochs using `CrossEntropyLoss` and the Adam optimizer (lr=0.001), saving the checkpoint with the best validation accuracy.

### STEP 6: Model Evaluation and Prediction
Evaluate the best checkpoint on the test set (loss, accuracy), visualize a confusion matrix and example predictions, and classify new, unseen images via a bonus prediction function / optional Streamlit app.

## PROGRAM

```python
import os, time, copy, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

# --- Environment / CUDA check ---
print("CUDA available:", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

SEED = 33
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

# --- Data preparation ---
DATA_DIR = "./data"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
CLASS_NAMES = ["cat", "dog", "panda"]
IMG_SIZE = 224
BATCH_SIZE = 32
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
test_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transforms)
test_dataset = datasets.ImageFolder(TEST_DIR, transform=test_transforms)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# --- Model design ---
def build_model(num_classes=3, freeze_backbone=True):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5),
        nn.Linear(256, num_classes),
    )
    return model

model = build_model(num_classes=len(CLASS_NAMES), freeze_backbone=True).to(device)

# --- Training ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
EPOCHS = 15
best_val_acc = 0.0
best_model_wts = copy.deepcopy(model.state_dict())

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return running_loss / total, 100.0 * correct / total, all_preds, all_labels

for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)

    train_loss = running_loss / len(train_dataset)
    val_loss, val_acc, _, _ = evaluate(model, test_loader, criterion, device)
    print(f"Epoch [{epoch}/{EPOCHS}]  Train Loss: {train_loss:.4f}  Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.2f}%")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_wts = copy.deepcopy(model.state_dict())
        torch.save(best_model_wts, "best_model.pth")

model.load_state_dict(best_model_wts)

# --- Evaluation ---
test_loss, test_acc, all_preds, all_labels = evaluate(model, test_loader, criterion, device)
print("Name: <YOUR NAME>")
print("Register No: <YOUR REGISTER NUMBER>")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.2f}%")
print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

cm = confusion_matrix(all_labels, all_preds)
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES).plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()

# --- Bonus: predict a new image ---
from PIL import Image

def predict_image(image_path, model=model, class_names=CLASS_NAMES, device=device):
    image = Image.open(image_path).convert("RGB")
    input_tensor = test_transforms(image).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1)
        pred_idx = logits.argmax(dim=1).item()
    print("Name: <YOUR NAME>")
    print("Register No: <YOUR REGISTER NUMBER>")
    print(f"Predicted class: {class_names[pred_idx]}")
    return class_names[pred_idx], probs.cpu().numpy()
```

### Dataset Information

| Split | Classes | Description |
|---|---|---|
| `train/cat`, `train/dog`, `train/panda` | 3 | Training images per class |
| `test/cat`, `test/dog`, `test/panda` | 3 | Held-out test images per class |

Source: [Cats vs Dogs vs Pandas — Kaggle](https://www.kaggle.com/datasets/gpiosenka/cats-dogs-pandas-images)

### OUTPUT

Fill in after running `notebooks/cat_dog_panda_transfer_learning.ipynb`:

- Training/validation loss & accuracy curves: *(insert plot/screenshot here)*
- Confusion matrix: *(insert plot/screenshot here)*
- Example predictions grid: *(insert plot/screenshot here)*
- Final test loss: *(insert value)*
- Final test accuracy: *(insert value)*

## RESULT

A transfer-learning image classification model (frozen ResNet18 backbone + custom fully-connected head) was trained successfully to classify images as cat, dog, or panda, and the reported test accuracy above confirms the model's performance on unseen data.

---

## Repository Structure

```
.
├── notebooks/
│   └── cat_dog_panda_transfer_learning.ipynb   # full implementation, training & results
├── data/                                        # dataset (download via Kaggle, see below)
│   ├── train/{cat,dog,panda}/
│   └── test/{cat,dog,panda}/
├── app.py                                       # optional Streamlit app (see notebook, bonus section)
├── requirements.txt
└── README.md
```

## Dataset

**Cats vs Dogs vs Pandas** — https://www.kaggle.com/datasets/gpiosenka/cats-dogs-pandas-images

Download it directly inside the notebook via the Kaggle CLI (requires a Kaggle
API token at `~/.kaggle/kaggle.json`):

```bash
pip install kaggle
kaggle datasets download -d gpiosenka/cats-dogs-pandas-images -p ./data --unzip
```

Then reorganize the extracted files into:

```
data/
├── train/
│   ├── cat/
│   ├── dog/
│   └── panda/
└── test/
    ├── cat/
    ├── dog/
    └── panda/
```

## CUDA / GPU Setup

Before training, verify GPU availability:

```python
import torch
print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
```

- **Local machine:** requires an NVIDIA GPU with the CUDA Toolkit and drivers
  installed. Follow the [PyTorch installation guide](https://pytorch.org/get-started/locally/)
  to install the matching `torch`/`torchvision` build for your CUDA version.
- **Kaggle:** go to *Settings → Accelerator → GPU* in the notebook editor to
  enable a GPU runtime, then install/upgrade dependencies if needed:
  ```bash
  pip install torch torchvision torchaudio --upgrade
  ```

## Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/<your-repo-name>.git
   cd <your-repo-name>
   ```
2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set up your Kaggle API token and download the dataset (see [Dataset](#dataset) above).
5. Launch Jupyter and run the notebook top to bottom:
   ```bash
   jupyter notebook notebooks/cat_dog_panda_transfer_learning.ipynb
   ```
6. (Optional) Run the bonus Streamlit app after training, once `best_model.pth` exists:
   ```bash
   streamlit run app.py
   ```
#   B u i l d i n g - a n - A I - C l a s s i f i e r - I d e n t i f y i n g - C a t s - D o g s - P a n d a s - w i t h - P y T o r c h - C o m p l e t i o n - r e q u i r e m e n t s  
 