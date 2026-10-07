# ============================================================
# ============================================================
# PROPOSED MODEL
# ACRIMA GLAUCOMA CLASSIFICATION
#
# PROPOSED COMPONENTS:
#   1. CLAHE preprocessing
#   2. Multi-block CNN
#   3. CBAM Channel Attention
#   4. CBAM Spatial Attention
#   5. Global Average Pooling
#   6. Dense layer
#   7. Dropout
#   8. Grad-CAM
#
# DATASET:
#   ACRIMA
#
# IMPORTANT:
#   U-Net architecture is included below, but genuine
#   optic-disc U-Net training requires optic-disc masks
#   or pretrained U-Net weights.
# ============================================================
# ============================================================


# ============================================================
# 1. INSTALL PACKAGES
# ============================================================

!pip install -q torch torchvision scikit-learn matplotlib seaborn pillow pandas opencv-python


# ============================================================
# 2. IMPORT LIBRARIES
# ============================================================

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

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
# 3. GOOGLE DRIVE
# ============================================================

from google.colab import drive

# If Drive is already mounted, this will remount it safely.
drive.mount(
    "/content/drive",
    force_remount=True
)


# ============================================================
# 4. CONFIGURATION
# ============================================================

# EXACT SAME DATASET PATH AS YOUR BASE MODEL

DATA_DIR = (
    "/content/drive/MyDrive/"
    "Database/Database/Images"
)

IMAGE_SIZE = 200

BATCH_SIZE = 16

EPOCHS = 50

LEARNING_RATE = 0.0001

DROPOUT_RATE = 0.5

FILTERS = [16, 32, 64]

THRESHOLD = 0.5

VALIDATION_RATIO = 0.20

SEED = 42


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("========================================")
print("PROPOSED MODEL CONFIGURATION")
print("========================================")

print("Device:", DEVICE)

print("Dataset:", DATA_DIR)

print("Image size:", IMAGE_SIZE)

print("Batch size:", BATCH_SIZE)

print("Epochs:", EPOCHS)

print("Learning rate:", LEARNING_RATE)

print("Filters:", FILTERS)

print("Dropout:", DROPOUT_RATE)

print("Threshold:", THRESHOLD)


if not os.path.exists(DATA_DIR):

    raise FileNotFoundError(
        f"\nDataset not found:\n{DATA_DIR}"
    )


# ============================================================
# 5. REPRODUCIBILITY
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
# 6. RESULTS DIRECTORY
# ============================================================

RESULTS_DIR = (
    "/content/drive/MyDrive/"
    "ACRIMA_CNN_Results"
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)

PROPOSED_RESULTS_DIR = os.path.join(
    RESULTS_DIR,
    "Proposed_Model"
)

os.makedirs(
    PROPOSED_RESULTS_DIR,
    exist_ok=True
)


# ============================================================
# 7. LOAD ACRIMA DATASET
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


for filename in sorted(
    os.listdir(DATA_DIR)
):

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
    #
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
        "No image files found."
    )


dataframe = pd.DataFrame(
    data
)


print("\n========================================")
print("DATASET SUMMARY")
print("========================================")

print(
    dataframe["class_name"].value_counts()
)

print(
    "\nTotal images:",
    len(dataframe)
)

print(
    "Glaucoma:",
    int(
        (dataframe["label"] == 1).sum()
    )
)

print(
    "Normal:",
    int(
        (dataframe["label"] == 0).sum()
    )
)


if dataframe["label"].nunique() < 2:

    raise ValueError(
        "Both normal and glaucoma classes are required."
    )


# ============================================================
# 8. SAME TRAIN / VALIDATION SPLIT AS BASE MODEL
# ============================================================

train_dataframe, validation_dataframe = (
    train_test_split(

        dataframe,

        test_size=VALIDATION_RATIO,

        random_state=SEED,

        stratify=dataframe["label"]
    )
)


train_dataframe = (
    train_dataframe
    .reset_index(drop=True)
)


validation_dataframe = (
    validation_dataframe
    .reset_index(drop=True)
)


print("\nTraining images:",
      len(train_dataframe))

print(
    "Validation images:",
    len(validation_dataframe)
)


# ============================================================
# 9. CLAHE FUNCTION
# ============================================================

def apply_clahe(image):

    # PIL -> numpy

    image = np.array(
        image
    )

    # RGB -> LAB

    lab = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2LAB
    )

    l_channel, a_channel, b_channel = (
        cv2.split(lab)
    )

    # CLAHE

    clahe = cv2.createCLAHE(

        clipLimit=2.0,

        tileGridSize=(8, 8)
    )

    l_channel = clahe.apply(
        l_channel
    )

    # Merge

    lab = cv2.merge(
        (
            l_channel,
            a_channel,
            b_channel
        )
    )

    # LAB -> RGB

    image = cv2.cvtColor(
        lab,
        cv2.COLOR_LAB2RGB
    )

    return Image.fromarray(
        image
    )


# ============================================================
# 10. CLAHE TRANSFORM
# ============================================================

class CLAHETransform:

    def __init__(
        self,
        train=False
    ):

        self.train = train


    def __call__(
        self,
        image
    ):

        image = apply_clahe(
            image
        )

        image = image.resize(
            (
                IMAGE_SIZE,
                IMAGE_SIZE
            )
        )

        if self.train:

            if random.random() < 0.5:

                image = image.transpose(
                    Image.Transpose.FLIP_LEFT_RIGHT
                )


        image = transforms.ToTensor()(
            image
        )

        image = transforms.Normalize(

            mean=[
                0.485,
                0.456,
                0.406
            ],

            std=[
                0.229,
                0.224,
                0.225
            ]
        )(image)

        return image


train_transform = CLAHETransform(
    train=True
)


validation_transform = CLAHETransform(
    train=False
)


# ============================================================
# 11. DATASET
# ============================================================

class ProposedACRIMADataset(
    Dataset
):

    def __init__(
        self,
        dataframe,
        transform=None
    ):

        self.dataframe = (
            dataframe
            .reset_index(drop=True)
        )

        self.transform = transform


    def __len__(self):

        return len(
            self.dataframe
        )


    def __getitem__(
        self,
        index
    ):

        image_path = (
            self.dataframe
            .loc[index, "filepath"]
        )

        label = (
            self.dataframe
            .loc[index, "label"]
        )


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
# 12. DATASETS
# ============================================================

train_dataset = (
    ProposedACRIMADataset(

        train_dataframe,

        train_transform
    )
)


validation_dataset = (
    ProposedACRIMADataset(

        validation_dataframe,

        validation_transform
    )
)


# ============================================================
# 13. DATA LOADERS
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
# 14. CBAM CHANNEL ATTENTION
# ============================================================

class ChannelAttention(
    nn.Module
):

    def __init__(
        self,
        channels,
        reduction=16
    ):

        super().__init__()


        reduced_channels = max(
            channels // reduction,
            1
        )


        self.avg_pool = (
            nn.AdaptiveAvgPool2d(1)
        )


        self.max_pool = (
            nn.AdaptiveMaxPool2d(1)
        )


        self.mlp = nn.Sequential(

            nn.Conv2d(
                channels,
                reduced_channels,
                kernel_size=1,
                bias=False
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                reduced_channels,
                channels,
                kernel_size=1,
                bias=False
            )
        )


        self.sigmoid = nn.Sigmoid()


    def forward(
        self,
        x
    ):

        avg_attention = self.mlp(
            self.avg_pool(x)
        )

        max_attention = self.mlp(
            self.max_pool(x)
        )

        attention = (
            avg_attention
            +
            max_attention
        )

        attention = self.sigmoid(
            attention
        )

        return (
            x * attention
        )


# ============================================================
# 15. CBAM SPATIAL ATTENTION
# ============================================================

class SpatialAttention(
    nn.Module
):

    def __init__(
        self,
        kernel_size=7
    ):

        super().__init__()


        padding = (
            kernel_size - 1
        ) // 2


        self.conv = nn.Conv2d(

            2,

            1,

            kernel_size=kernel_size,

            padding=padding,

            bias=False
        )


        self.sigmoid = nn.Sigmoid()


    def forward(
        self,
        x
    ):

        avg_attention = (
            torch.mean(
                x,
                dim=1,
                keepdim=True
            )
        )


        max_attention, _ = (
            torch.max(
                x,
                dim=1,
                keepdim=True
            )
        )


        combined = torch.cat(

            [
                avg_attention,
                max_attention
            ],

            dim=1
        )


        attention = self.conv(
            combined
        )


        attention = self.sigmoid(
            attention
        )


        return (
            x * attention
        )


# ============================================================
# 16. COMPLETE CBAM
# ============================================================

class CBAM(
    nn.Module
):

    def __init__(
        self,
        channels,
        reduction=16
    ):

        super().__init__()


        self.channel_attention = (
            ChannelAttention(
                channels,
                reduction
            )
        )


        self.spatial_attention = (
            SpatialAttention(
                kernel_size=7
            )
        )


    def forward(
        self,
        x
    ):

        x = self.channel_attention(
            x
        )

        x = self.spatial_attention(
            x
        )

        return x


# ============================================================
# 17. PROPOSED MULTI-BLOCK CNN
# ============================================================

class ProposedCNN(
    nn.Module
):

    def __init__(
        self,
        filters=(32, 64, 128),
        dropout_rate=0.5
    ):

        super().__init__()


        # ==========================================
        # BLOCK 1
        # ==========================================

        self.block1 = nn.Sequential(

            nn.Conv2d(
                3,
                filters[0],
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                filters[0]
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                filters[0],
                filters[0],
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                filters[0]
            ),

            nn.ReLU(inplace=True)
        )


        self.cbam1 = CBAM(
            filters[0]
        )


        self.pool1 = nn.MaxPool2d(
            2
        )


        # ==========================================
        # BLOCK 2
        # ==========================================

        self.block2 = nn.Sequential(

            nn.Conv2d(
                filters[0],
                filters[1],
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                filters[1]
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                filters[1],
                filters[1],
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                filters[1]
            ),

            nn.ReLU(inplace=True)
        )


        self.cbam2 = CBAM(
            filters[1]
        )


        self.pool2 = nn.MaxPool2d(
            2
        )


        # ==========================================
        # BLOCK 3
        # ==========================================

        self.block3 = nn.Sequential(

            nn.Conv2d(
                filters[1],
                filters[2],
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                filters[2]
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                filters[2],
                filters[2],
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                filters[2]
            ),

            nn.ReLU(inplace=True)
        )


        self.cbam3 = CBAM(
            filters[2]
        )


        self.pool3 = nn.MaxPool2d(
            2
        )


        # ==========================================
        # BLOCK 4
        # ==========================================

        self.block4 = nn.Sequential(

            nn.Conv2d(
                filters[2],
                filters[2],
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                filters[2]
            ),

            nn.ReLU(inplace=True)
        )


        self.cbam4 = CBAM(
            filters[2]
        )


        # ==========================================
        # GLOBAL AVERAGE POOLING
        # ==========================================

        self.global_average_pooling = (
            nn.AdaptiveAvgPool2d(
                (1, 1)
            )
        )


        # ==========================================
        # CLASSIFIER
        # ==========================================

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                filters[2],
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


    def forward(
        self,
        x
    ):

        # Block 1

        x = self.block1(x)

        x = self.cbam1(x)

        x = self.pool1(x)


        # Block 2

        x = self.block2(x)

        x = self.cbam2(x)

        x = self.pool2(x)


        # Block 3

        x = self.block3(x)

        x = self.cbam3(x)

        x = self.pool3(x)


        # Block 4

        x = self.block4(x)

        x = self.cbam4(x)


        # GAP

        x = self.global_average_pooling(
            x
        )


        # Classifier

        x = self.classifier(
            x
        )


        return x


# ============================================================
# 18. CREATE PROPOSED MODEL
# ============================================================

model = ProposedCNN(

    filters=FILTERS,

    dropout_rate=DROPOUT_RATE

).to(DEVICE)


print("\n========================================")
print("PROPOSED CNN")
print("========================================")

print(model)


# ============================================================
# 19. LOSS + OPTIMIZER
# ============================================================

criterion = nn.BCEWithLogitsLoss()


optimizer = optim.Adam(

    model.parameters(),

    lr=LEARNING_RATE
)


# ============================================================
# 20. TRAINING FUNCTION
# ============================================================

def train_one_epoch(

    model,

    loader,

    criterion,

    optimizer,

    device

):

    model.train()


    total_loss = 0.0

    total_correct = 0

    total_samples = 0


    for images, labels in loader:

        images = images.to(
            device
        )

        labels = labels.to(
            device
        )

        labels = labels.view(
            -1,
            1
        )


        optimizer.zero_grad()


        logits = model(
            images
        )


        loss = criterion(
            logits,
            labels
        )


        loss.backward()


        optimizer.step()


        total_loss += (
            loss.item()
            *
            images.size(0)
        )


        probabilities = (
            torch.sigmoid(
                logits
            )
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


    return (
        average_loss,
        accuracy
    )


# ============================================================
# 21. VALIDATION FUNCTION
# ============================================================

def validate_model(

    model,

    loader,

    criterion,

    device

):

    model.eval()


    total_loss = 0.0

    total_correct = 0

    total_samples = 0


    all_labels = []

    all_probabilities = []


    with torch.no_grad():

        for images, labels in loader:

            images = images.to(
                device
            )

            labels = labels.to(
                device
            )

            labels = labels.view(
                -1,
                1
            )


            logits = model(
                images
            )


            loss = criterion(
                logits,
                labels
            )


            total_loss += (
                loss.item()
                *
                images.size(0)
            )


            probabilities = (
                torch.sigmoid(
                    logits
                )
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

                labels
                .cpu()
                .numpy()
                .ravel()
            )


            all_probabilities.extend(

                probabilities
                .cpu()
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

        np.array(
            all_labels
        ).astype(int),

        np.array(
            all_probabilities
        )
    )


# ============================================================
# 22. TRAIN PROPOSED MODEL
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

    PROPOSED_RESULTS_DIR,

    "proposed_cnn_best.pth"
)


print("\n========================================")
print("PROPOSED MODEL TRAINING")
print("========================================")


for epoch in range(
    EPOCHS
):


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
    ].append(
        train_loss
    )


    history[
        "train_accuracy"
    ].append(
        train_accuracy
    )


    history[
        "validation_loss"
    ].append(
        validation_loss
    )


    history[
        "validation_accuracy"
    ].append(
        validation_accuracy
    )


    print(

        f"Epoch [{epoch+1:03d}/{EPOCHS}] | "

        f"Train Loss: {train_loss:.4f} | "

        f"Train Acc: {train_accuracy:.4f} | "

        f"Val Loss: {validation_loss:.4f} | "

        f"Val Acc: {validation_accuracy:.4f}"

    )


    # ------------------------------------------
    # SAVE BEST MODEL
    # ------------------------------------------

    if (
        validation_loss
        <
        best_validation_loss
    ):

        best_validation_loss = (
            validation_loss
        )


        torch.save(

            {

                "epoch":
                    epoch + 1,

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
                    THRESHOLD,

                "clahe":
                    True,

                "cbam":
                    True

            },

            best_model_path
        )


print(
    "\nProposed model training completed."
)


# ============================================================
# 23. LOAD BEST MODEL
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
# 24. FINAL VALIDATION PREDICTIONS
# ============================================================

all_true_labels = []

all_probabilities = []


with torch.no_grad():

    for images, labels in validation_loader:

        images = images.to(
            DEVICE
        )


        logits = model(
            images
        )


        probabilities = (
            torch.sigmoid(
                logits
            )
        )


        all_true_labels.extend(

            labels
            .numpy()
            .ravel()
        )


        all_probabilities.extend(

            probabilities
            .cpu()
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

    all_probabilities
    >=
    THRESHOLD

).astype(int)


# ============================================================
# 25. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(

    all_true_labels,

    all_predictions,

    labels=[0, 1]
)


tn, fp, fn, tp = cm.ravel()


# ============================================================
# 26. METRICS
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
# 27. PRINT RESULTS
# ============================================================

print("\n========================================")
print("PROPOSED MODEL FINAL RESULTS")
print("========================================")


print("\nLabel mapping:")

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
# 28. SAVE METRICS
# ============================================================

metrics_file = os.path.join(

    PROPOSED_RESULTS_DIR,

    "proposed_model_metrics.txt"
)


with open(

    metrics_file,

    "w"

) as file:


    file.write(
        "PROPOSED CNN RESULTS\n"
    )

    file.write(
        "=====================\n\n"
    )

    file.write(
        "Dataset: ACRIMA\n"
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
        f"Dropout: {DROPOUT_RATE}\n"
    )

    file.write(
        f"Threshold: {THRESHOLD}\n"
    )

    file.write(
        "CLAHE: True\n"
    )

    file.write(
        "CBAM: True\n"
    )

    file.write(
        "U-Net ROI: Not activated\n"
    )

    file.write(
        "Grad-CAM: Available\n\n"
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


# ============================================================
# 29. CONFUSION MATRIX
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
    "Proposed CNN Confusion Matrix"
)


plt.tight_layout()


cm_path = os.path.join(

    PROPOSED_RESULTS_DIR,

    "proposed_confusion_matrix.png"
)


plt.savefig(

    cm_path,

    dpi=300
)


plt.show()


# ============================================================
# 30. TRAINING CURVES
# ============================================================

epochs_range = range(

    1,

    EPOCHS + 1
)


plt.figure(

    figsize=(12, 5)
)


# ------------------------------------------
# ACCURACY
# ------------------------------------------

plt.subplot(

    1,

    2,

    1
)


plt.plot(

    epochs_range,

    history[
        "train_accuracy"
    ],

    label="Training"
)


plt.plot(

    epochs_range,

    history[
        "validation_accuracy"
    ],

    label="Validation"
)


plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Accuracy"
)

plt.title(
    "Proposed Model Accuracy"
)

plt.legend()


# ------------------------------------------
# LOSS
# ------------------------------------------

plt.subplot(

    1,

    2,

    2
)


plt.plot(

    epochs_range,

    history[
        "train_loss"
    ],

    label="Training"
)


plt.plot(

    epochs_range,

    history[
        "validation_loss"
    ],

    label="Validation"
)


plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Loss"
)

plt.title(
    "Proposed Model Loss"
)

plt.legend()


plt.tight_layout()


history_path = os.path.join(

    PROPOSED_RESULTS_DIR,

    "proposed_training_history.png"
)


plt.savefig(

    history_path,

    dpi=300
)


plt.show()


# ============================================================
# 31. GRAD-CAM
# ============================================================

class GradCAM:

    def __init__(
        self,
        model,
        target_layer
    ):

        self.model = model

        self.target_layer = (
            target_layer
        )

        self.activations = None

        self.gradients = None


        self.forward_hook = (
            target_layer.register_forward_hook(
                self.save_activation
            )
        )


        self.backward_hook = (
            target_layer.register_full_backward_hook(
                self.save_gradient
            )
        )


    def save_activation(
        self,
        module,
        input,
        output
    ):

        self.activations = output


    def save_gradient(
        self,
        module,
        grad_input,
        grad_output
    ):

        self.gradients = (
            grad_output[0]
        )


    def generate(
        self,
        image
    ):

        self.model.eval()

        self.model.zero_grad()


        output = self.model(
            image
        )


        score = output[:, 0]


        score.backward()


        gradients = (
            self.gradients
        )


        activations = (
            self.activations
        )


        weights = torch.mean(

            gradients,

            dim=(2, 3),

            keepdim=True
        )


        cam = torch.sum(

            weights
            *
            activations,

            dim=1,

            keepdim=True
        )


        cam = torch.relu(
            cam
        )


        cam = torch.nn.functional.interpolate(

            cam,

            size=(
                IMAGE_SIZE,
                IMAGE_SIZE
            ),

            mode="bilinear",

            align_corners=False
        )


        cam = cam.squeeze()

        cam = cam.detach().cpu().numpy()


        cam = (
            cam - cam.min()
        ) / (
            cam.max()
            -
            cam.min()
            +
            1e-8
        )


        return cam


    def close(self):

        self.forward_hook.remove()

        self.backward_hook.remove()


# ============================================================
# 32. GRAD-CAM TARGET LAYER
# ============================================================

target_layer = model.block4[0]


gradcam = GradCAM(

    model,

    target_layer
)


# ============================================================
# 33. SELECT ONE VALIDATION IMAGE
# ============================================================

sample_image, sample_label = (
    validation_dataset[0]
)


sample_input = (
    sample_image
    .unsqueeze(0)
    .to(DEVICE)
)


# ============================================================
# 34. GENERATE GRAD-CAM
# ============================================================

cam = gradcam.generate(

    sample_input
)


# ============================================================
# 35. LOAD ORIGINAL IMAGE
# ============================================================

sample_path = (
    validation_dataframe
    .iloc[0]["filepath"]
)


original_image = Image.open(

    sample_path

).convert("RGB")


original_image = original_image.resize(

    (
        IMAGE_SIZE,
        IMAGE_SIZE
    )
)


original_array = np.array(
    original_image
)


# ============================================================
# 36. CREATE HEATMAP
# ============================================================

heatmap = np.uint8(
    255 * cam
)


heatmap = cv2.applyColorMap(

    heatmap,

    cv2.COLORMAP_JET
)


heatmap = cv2.cvtColor(

    heatmap,

    cv2.COLOR_BGR2RGB
)


overlay = cv2.addWeighted(

    original_array,

    0.6,

    heatmap,

    0.4,

    0
)


# ============================================================
# 37. DISPLAY GRAD-CAM
# ============================================================

plt.figure(
    figsize=(12, 4)
)


plt.subplot(
    1,
    3,
    1
)

plt.imshow(
    original_array
)

plt.title(
    "Original Fundus"
)

plt.axis(
    "off"
)


plt.subplot(
    1,
    3,
    2
)

plt.imshow(
    heatmap
)

plt.title(
    "Grad-CAM"
)

plt.axis(
    "off"
)


plt.subplot(
    1,
    3,
    3
)

plt.imshow(
    overlay
)

plt.title(
    "Grad-CAM Overlay"
)

plt.axis(
    "off"
)


plt.tight_layout()


gradcam_path = os.path.join(

    PROPOSED_RESULTS_DIR,

    "gradcam_example.png"
)


plt.savefig(

    gradcam_path,

    dpi=300
)


plt.show()


gradcam.close()


# ============================================================
# 38. SAVE FINAL PROPOSED MODEL
# ============================================================

final_model_path = os.path.join(

    PROPOSED_RESULTS_DIR,

    "proposed_cnn_final.pth"
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
            THRESHOLD,

        "dropout":
            DROPOUT_RATE,

        "clahe":
            True,

        "cbam":
            True,

        "unet_roi":
            False,

        "gradcam":
            True

    },

    final_model_path
)


# ============================================================
# 39. SAVE RESULT CSV
# ============================================================

result_dataframe = pd.DataFrame({

    "filename":
        validation_dataframe[
            "filename"
        ].values,

    "true_label":
        all_true_labels,

    "predicted_probability":
        all_probabilities,

    "predicted_label":
        all_predictions

})


prediction_csv = os.path.join(

    PROPOSED_RESULTS_DIR,

    "proposed_validation_predictions.csv"
)


result_dataframe.to_csv(

    prediction_csv,

    index=False
)


# ============================================================
# 40. FINAL OUTPUT
# ============================================================

print("\n")
print("========================================")
print("PROPOSED MODEL COMPLETED")
print("========================================")

print(
    "\nAccuracy:",
    f"{accuracy:.4f}"
)

print(
    "Sensitivity:",
    f"{sensitivity:.4f}"
)

print(
    "Specificity:",
    f"{specificity:.4f}"
)

print(
    "Precision:",
    f"{precision:.4f}"
)

print(
    "F1:",
    f"{f1:.4f}"
)

print(
    "ROC-AUC:",
    f"{auc:.4f}"
)


print("\nSaved files:")

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
    "Confusion matrix:",
    cm_path
)

print(
    "Training curves:",
    history_path
)

print(
    "Grad-CAM:",
    gradcam_path
)

print(
    "Predictions:",
    prediction_csv
)

print(
    "\nResults directory:",
    PROPOSED_RESULTS_DIR
)
