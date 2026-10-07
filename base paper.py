# ============================================================
# CELL 1 — MOUNT GOOGLE DRIVE
# ============================================================

from google.colab import drive
import os

drive.mount("/content/drive", force_remount=True)

DATA_DIR = "/content/drive/MyDrive/Database/Database/Images"

print("Drive mounted:", os.path.exists("/content/drive"))
print("Dataset exists:", os.path.exists(DATA_DIR))

if os.path.exists(DATA_DIR):
    print("\nFiles/Folders inside Images:")
    print(os.listdir(DATA_DIR)[:20])
else:
    raise FileNotFoundError(
        f"Dataset not found:\n{DATA_DIR}"
    )
# ============================================================
# ACRIMA BASE-PAPER CNN
# PYTORCH
# ============================================================

!pip install -q torch torchvision scikit-learn matplotlib seaborn pillow pandas

# ============================================================
# 1. IMPORTS
# ============================================================

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# ============================================================
# 2. CONFIGURATION
# ============================================================

DATA_DIR = "/content/drive/MyDrive/Database/Database/Images"

IMAGE_SIZE = 200
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 0.0001
DROPOUT_RATE = 0.5

# YOUR BASE MODEL
FILTERS = [32, 64, 128]

THRESHOLD = 0.5
VALIDATION_RATIO = 0.05
SEED = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", DEVICE)
print("Dataset path:", DATA_DIR)

if not os.path.exists(DATA_DIR):
    raise FileNotFoundError(
        f"Dataset folder not found:\n{DATA_DIR}"
    )

# ============================================================
# 3. REPRODUCIBILITY
# ============================================================

def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(SEED)

# ============================================================
# 4. RESULTS DIRECTORY
# ============================================================

RESULTS_DIR = (
    "/content/drive/MyDrive/"
    "ACRIMA_CNN_Results"
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)

# ============================================================
# 5. LOAD ACRIMA
# ============================================================

valid_extensions = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff"
)

data = []

for filename in sorted(os.listdir(DATA_DIR)):

    if not filename.lower().endswith(
        valid_extensions
    ):
        continue

    filepath = os.path.join(
        DATA_DIR,
        filename
    )

    # -----------------------------------------
    # ACRIMA LABEL RULE
    # "_g_" = glaucoma
    # otherwise = normal
    # -----------------------------------------

    if "_g_" in filename.lower():

        label = 1
        class_name = "glaucoma"

    else:

        label = 0
        class_name = "normal"

    data.append({
        "filepath": filepath,
        "filename": filename,
        "label": label,
        "class_name": class_name
    })


if len(data) == 0:

    raise ValueError(
        "No images found in DATA_DIR."
    )


dataframe = pd.DataFrame(data)

print("\n==============================")
print("ACRIMA DATASET")
print("==============================")

print(
    dataframe["class_name"].value_counts()
)

print(
    "\nTotal images:",
    len(dataframe)
)

print(
    "Glaucoma:",
    (dataframe["label"] == 1).sum()
)

print(
    "Normal:",
    (dataframe["label"] == 0).sum()
)

# ============================================================
# 6. TRAIN / VALIDATION SPLIT
# ============================================================

train_dataframe, validation_dataframe = train_test_split(

    dataframe,

    test_size=VALIDATION_RATIO,

    random_state=SEED,

    stratify=dataframe["label"]
)

train_dataframe = train_dataframe.reset_index(
    drop=True
)

validation_dataframe = validation_dataframe.reset_index(
    drop=True
)

print("\nTraining images:",
      len(train_dataframe))

print("Validation images:",
      len(validation_dataframe))

# ============================================================
# 7. TRANSFORMS
# ============================================================

train_transforms = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomRotation(
        degrees=10
    ),

    transforms.RandomAffine(
        degrees=0,
        translate=(0.05, 0.05),
        scale=(0.95, 1.05)
    ),

    transforms.ColorJitter(
        brightness=0.10,
        contrast=0.10
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


validation_transforms = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ============================================================
# 8. DATASET CLASS
# ============================================================

class ACRIMADataset(Dataset):

    def __init__(
        self,
        dataframe,
        transform=None
    ):

        self.dataframe = dataframe.reset_index(
            drop=True
        )

        self.transform = transform


    def __len__(self):

        return len(self.dataframe)


    def __getitem__(self, index):

        image_path = self.dataframe.loc[
            index,
            "filepath"
        ]

        label = self.dataframe.loc[
            index,
            "label"
        ]

        image = Image.open(
            image_path
        ).convert("RGB")

        if self.transform:

            image = self.transform(
                image
            )

        label = torch.tensor(
            label,
            dtype=torch.float32
        )

        return image, label


# ============================================================
# 9. DATASETS
# ============================================================

train_dataset = ACRIMADataset(
    train_dataframe,
    train_transforms
)

validation_dataset = ACRIMADataset(
    validation_dataframe,
    validation_transforms
)

# ============================================================
# 10. DATA LOADERS
# ============================================================

train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True,

    num_workers=2,

    pin_memory=torch.cuda.is_available()
)


validation_loader = DataLoader(

    validation_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=2,

    pin_memory=torch.cuda.is_available()
)

# ============================================================
# 11. BASE-PAPER CNN
# ============================================================

class BasePaperCNN(nn.Module):

    def __init__(
        self,
        filters=[32, 64, 128],
        dropout_rate=0.5
    ):

        super().__init__()

        self.features = nn.Sequential(

            # -------------------------
            # BLOCK 1
            # -------------------------

            nn.Conv2d(
                3,
                filters[0],
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),

            # -------------------------
            # BLOCK 2
            # -------------------------

            nn.Conv2d(
                filters[0],
                filters[1],
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),

            # -------------------------
            # BLOCK 3
            # -------------------------

            nn.Conv2d(
                filters[1],
                filters[2],
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),

            nn.Dropout(
                dropout_rate
            )
        )

        # 200 -> 100 -> 50 -> 25

        flattened_features = (
            filters[2] * 25 * 25
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                flattened_features,
                128
            ),

            nn.ReLU(inplace=True),

            nn.Dropout(
                dropout_rate
            ),

            nn.Linear(
                128,
                1
            )
        )


    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x


model = BasePaperCNN(
    filters=FILTERS,
    dropout_rate=DROPOUT_RATE
).to(DEVICE)

print("\n==============================")
print("BASE MODEL")
print("==============================")

print(model)

# ============================================================
# 12. LOSS + OPTIMIZER
# ============================================================

criterion = nn.BCEWithLogitsLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

# ============================================================
# 13. TRAIN FUNCTION
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device
):

    model.train()

    total_loss = 0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:

        images = images.to(device)

        labels = labels.to(device)

        labels = labels.view(-1, 1)

        optimizer.zero_grad()

        logits = model(images)

        loss = criterion(
            logits,
            labels
        )

        loss.backward()

        optimizer.step()

        total_loss += (
            loss.item()
            * images.size(0)
        )

        probabilities = torch.sigmoid(
            logits
        )

        predictions = (
            probabilities >= 0.5
        ).float()

        total_correct += (
            predictions == labels
        ).sum().item()

        total_samples += (
            labels.size(0)
        )

    average_loss = (
        total_loss /
        total_samples
    )

    accuracy = (
        total_correct /
        total_samples
    )

    return average_loss, accuracy

# ============================================================
# 14. VALIDATION FUNCTION
# ============================================================

def validate_model(
    model,
    loader,
    criterion,
    device
):

    model.eval()

    total_loss = 0
    total_correct = 0
    total_samples = 0

    all_labels = []
    all_probabilities = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)

            labels = labels.to(device)

            labels = labels.view(-1, 1)

            logits = model(images)

            loss = criterion(
                logits,
                labels
            )

            total_loss += (
                loss.item()
                * images.size(0)
            )

            probabilities = torch.sigmoid(
                logits
            )

            predictions = (
                probabilities >= 0.5
            ).float()

            total_correct += (
                predictions == labels
            ).sum().item()

            total_samples += (
                labels.size(0)
            )

            all_labels.extend(
                labels.cpu()
                .numpy()
                .ravel()
            )

            all_probabilities.extend(
                probabilities.cpu()
                .numpy()
                .ravel()
            )

    average_loss = (
        total_loss /
        total_samples
    )

    accuracy = (
        total_correct /
        total_samples
    )

    return (
        average_loss,
        accuracy,
        np.array(all_labels).astype(int),
        np.array(all_probabilities)
    )

# ============================================================
# 15. TRAIN
# ============================================================

history = {

    "train_loss": [],
    "train_accuracy": [],

    "validation_loss": [],
    "validation_accuracy": []
}

best_validation_loss = float(
    "inf"
)

best_model_path = os.path.join(

    RESULTS_DIR,

    "base_paper_cnn_best.pth"
)

print("\n==============================")
print("TRAINING STARTED")
print("==============================")

for epoch in range(EPOCHS):

    train_loss, train_accuracy = (
        train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            DEVICE
        )
    )

    validation_loss, validation_accuracy, _, _ = (
        validate_model(
            model,
            validation_loader,
            criterion,
            DEVICE
        )
    )

    history[
        "train_loss"
    ].append(train_loss)

    history[
        "train_accuracy"
    ].append(train_accuracy)

    history[
        "validation_loss"
    ].append(validation_loss)

    history[
        "validation_accuracy"
    ].append(validation_accuracy)

    print(
        f"Epoch [{epoch+1:03d}/{EPOCHS}] | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy:.4f} | "
        f"Val Loss: {validation_loss:.4f} | "
        f"Val Acc: {validation_accuracy:.4f}"
    )

    # Save best validation model

    if validation_loss < best_validation_loss:

        best_validation_loss = (
            validation_loss
        )

        torch.save(

            {
                "epoch": epoch + 1,

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "validation_loss":
                    validation_loss,

                "filters":
                    FILTERS,

                "image_size":
                    IMAGE_SIZE,

                "threshold":
                    THRESHOLD
            },

            best_model_path
        )

print("\nTraining completed.")

# ============================================================
# 16. LOAD BEST MODEL
# ============================================================

checkpoint = torch.load(
    best_model_path,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint[
        "model_state_dict"
    ]
)

model.eval()

# ============================================================
# 17. FINAL VALIDATION PREDICTIONS
# ============================================================

all_true_labels = []
all_probabilities = []

with torch.no_grad():

    for images, labels in validation_loader:

        images = images.to(DEVICE)

        logits = model(images)

        probabilities = torch.sigmoid(
            logits
        )

        all_true_labels.extend(
            labels.numpy().ravel()
        )

        all_probabilities.extend(
            probabilities.cpu()
            .numpy()
            .ravel()
        )


all_true_labels = np.array(
    all_true_labels
).astype(int)

all_probabilities = np.array(
    all_probabilities
)

all_predictions = (
    all_probabilities >= THRESHOLD
).astype(int)

# ============================================================
# 18. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(

    all_true_labels,

    all_predictions,

    labels=[0, 1]
)

tn, fp, fn, tp = cm.ravel()

# ============================================================
# 19. METRICS
# ============================================================

accuracy = accuracy_score(
    all_true_labels,
    all_predictions
)

precision = precision_score(
    all_true_labels,
    all_predictions,
    zero_division=0
)

sensitivity = recall_score(
    all_true_labels,
    all_predictions,
    zero_division=0
)

specificity = (

    tn / (tn + fp)

    if (tn + fp) > 0

    else 0
)

f1 = f1_score(
    all_true_labels,
    all_predictions,
    zero_division=0
)

try:

    auc = roc_auc_score(
        all_true_labels,
        all_probabilities
    )

except ValueError:

    auc = float("nan")

# ============================================================
# 20. PRINT RESULTS
# ============================================================

print("\n")
print("========================================")
print("BASE-PAPER CNN FINAL RESULTS")
print("========================================")

print("0 = Normal")
print("1 = Glaucoma")

print("\nConfusion Matrix:")
print(cm)

print("\nTP:", tp)
print("TN:", tn)
print("FP:", fp)
print("FN:", fn)

print("\nMetrics:")

print(
    f"Accuracy:    {accuracy:.4f}"
)

print(
    f"Sensitivity: {sensitivity:.4f}"
)

print(
    f"Specificity: {specificity:.4f}"
)

print(
    f"Precision:   {precision:.4f}"
)

print(
    f"F1-score:    {f1:.4f}"
)

print(
    f"ROC-AUC:     {auc:.4f}"
)

print("\nClassification Report:")

print(
    classification_report(
        all_true_labels,
        all_predictions,
        target_names=[
            "normal",
            "glaucoma"
        ],
        zero_division=0
    )
)

# ============================================================
# 21. SAVE METRICS
# ============================================================

metrics_file = os.path.join(

    RESULTS_DIR,

    "base_model_metrics.txt"
)

with open(
    metrics_file,
    "w"
) as file:

    file.write(
        "BASE-PAPER CNN RESULTS\n"
    )

    file.write(
        "========================\n\n"
    )

    file.write(
        f"Dataset: ACRIMA\n"
    )

    file.write(
        f"Image size: {IMAGE_SIZE}\n"
    )

    file.write(
        f"Filters: {FILTERS}\n"
    )

    file.write(
        f"Batch size: {BATCH_SIZE}\n"
    )

    file.write(
        f"Epochs: {EPOCHS}\n"
    )

    file.write(
        f"Learning rate: {LEARNING_RATE}\n"
    )

    file.write(
        f"Threshold: {THRESHOLD}\n\n"
    )

    file.write(
        f"TP: {tp}\n"
    )

    file.write(
        f"TN: {tn}\n"
    )

    file.write(
        f"FP: {fp}\n"
    )

    file.write(
        f"FN: {fn}\n\n"
    )

    file.write(
        f"Accuracy: {accuracy:.4f}\n"
    )

    file.write(
        f"Sensitivity: {sensitivity:.4f}\n"
    )

    file.write(
        f"Specificity: {specificity:.4f}\n"
    )

    file.write(
        f"Precision: {precision:.4f}\n"
    )

    file.write(
        f"F1-score: {f1:.4f}\n"
    )

    file.write(
        f"ROC-AUC: {auc:.4f}\n"
    )

print(
    "\nMetrics saved to:",
    metrics_file
)

# ============================================================
# 22. CONFUSION MATRIX PLOT
# ============================================================

plt.figure(
    figsize=(6, 5)
)

sns.heatmap(

    cm,

    annot=True,

    fmt="d",

    cmap="Blues",

    xticklabels=[
        "Normal",
        "Glaucoma"
    ],

    yticklabels=[
        "Normal",
        "Glaucoma"
    ]
)

plt.xlabel(
    "Predicted"
)

plt.ylabel(
    "Actual"
)

plt.title(
    "Base-Paper CNN Confusion Matrix"
)

plt.tight_layout()

cm_path = os.path.join(

    RESULTS_DIR,

    "base_confusion_matrix.png"
)

plt.savefig(
    cm_path,
    dpi=300
)

plt.show()

# ============================================================
# 23. TRAINING CURVES
# ============================================================

epochs_range = range(
    1,
    EPOCHS + 1
)

plt.figure(
    figsize=(12, 5)
)

# Accuracy

plt.subplot(
    1,
    2,
    1
)

plt.plot(
    epochs_range,
    history["train_accuracy"],
    label="Training"
)

plt.plot(
    epochs_range,
    history["validation_accuracy"],
    label="Validation"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.title(
    "Training vs Validation Accuracy"
)

plt.legend()

# Loss

plt.subplot(
    1,
    2,
    2
)

plt.plot(
    epochs_range,
    history["train_loss"],
    label="Training"
)

plt.plot(
    epochs_range,
    history["validation_loss"],
    label="Validation"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title(
    "Training vs Validation Loss"
)

plt.legend()

plt.tight_layout()

history_path = os.path.join(

    RESULTS_DIR,

    "base_training_history.png"
)

plt.savefig(
    history_path,
    dpi=300
)

plt.show()

# ============================================================
# 24. SAVE FINAL MODEL
# ============================================================

final_model_path = os.path.join(

    RESULTS_DIR,

    "base_paper_cnn_final.pth"
)

torch.save(

    {
        "model_state_dict":
            model.state_dict(),

        "filters":
            FILTERS,

        "image_size":
            IMAGE_SIZE,

        "threshold":
            THRESHOLD
    },

    final_model_path
)

# ============================================================
# 25. FINAL OUTPUT
# ============================================================

print("\n========================================")
print("BASE MODEL COMPLETED")
print("========================================")

print(
    "Best model:",
    best_model_path
)

print(
    "Final model:",
    final_model_path
)

print(
    "Metrics:",
    metrics_file
)

print(
    "Results directory:",
    RESULTS_DIR
)
