import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from torch.utils.data import DataLoader
from torchvision import models, transforms


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_NAME = (
    "hf-tuner/rvl-cdip-document-classification"
)

MODEL_PATH = (
    "models/document_classifier_v2.pth"
)

CLASS_NAMES_PATH = (
    "models/class_names.json"
)

BATCH_SIZE = 32


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 60)
print("DocuMind AI — Model Evaluation")
print("=" * 60)

print(f"Device: {device}")

if torch.cuda.is_available():

    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )


# ============================================================
# LOAD CLASS NAMES
# ============================================================

with open(
    CLASS_NAMES_PATH,
    "r"
) as file:

    class_names = json.load(file)


print(
    f"\nNumber of classes: {len(class_names)}"
)


# ============================================================
# LOAD DATASET
# ============================================================

print("\nLoading dataset...")

dataset = load_dataset(
    DATASET_NAME
)

test_dataset = dataset["test"]


print(
    f"Evaluation images: "
    f"{len(test_dataset)}"
)


# ============================================================
# VALIDATION TRANSFORM
# ============================================================

transform = transforms.Compose([

    transforms.Resize(
        (224, 224)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406,
        ],
        std=[
            0.229,
            0.224,
            0.225,
        ],
    ),
])


# ============================================================
# DATASET TRANSFORMATION
# ============================================================

def transform_images(example):

    example["image"] = [
        transform(
            image.convert("RGB")
        )
        for image in example["image"]
    ]

    return example


test_dataset = test_dataset.with_transform(
    transform_images
)


# ============================================================
# DATALOADER
# ============================================================

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)


# ============================================================
# CREATE V2 MODEL
# ============================================================

print("\nLoading V2 model...")

model = models.resnet18(
    weights=None
)

model.fc = nn.Sequential(

    nn.Dropout(
        p=0.3
    ),

    nn.Linear(
        model.fc.in_features,
        len(class_names),
    ),
)


# ============================================================
# LOAD WEIGHTS
# ============================================================

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True,
    )
)


model = model.to(device)

model.eval()


print(
    "✓ V2 model loaded successfully."
)


# ============================================================
# MODEL PREDICTIONS
# ============================================================

all_predictions = []

all_labels = []


print("\nRunning evaluation...")


with torch.no_grad():

    for batch in test_loader:

        images = batch[
            "image"
        ].to(
            device,
            non_blocking=True,
        )

        labels = batch[
            "label"
        ].to(
            device,
            non_blocking=True,
        )


        outputs = model(
            images
        )


        predictions = outputs.argmax(
            dim=1
        )


        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.cpu().numpy()
        )


# ============================================================
# OVERALL ACCURACY
# ============================================================

accuracy = accuracy_score(
    all_labels,
    all_predictions,
)


print("\n" + "=" * 60)

print("OVERALL RESULTS")

print("=" * 60)

print(
    f"Accuracy: {accuracy * 100:.2f}%"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 60)

print("CLASSIFICATION REPORT")

print("=" * 60)


report = classification_report(
    all_labels,
    all_predictions,
    target_names=class_names,
    digits=4,
)


print(report)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    all_labels,
    all_predictions,
)


print("\nConfusion matrix generated.")


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

Path(
    "models/evaluation"
).mkdir(
    parents=True,
    exist_ok=True,
)


fig, ax = plt.subplots(
    figsize=(14, 14)
)


display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names,
)


display.plot(
    ax=ax,
    xticks_rotation=90,
    colorbar=False,
)


plt.title(
    "DocuMind AI — ResNet18 V2 Confusion Matrix"
)


plt.tight_layout()


plt.savefig(
    "models/evaluation/confusion_matrix.png",
    dpi=200,
    bbox_inches="tight",
)


plt.close()


print(
    "\nSaved:"
)

print(
    "models/evaluation/confusion_matrix.png"
)


# ============================================================
# SAVE CLASSIFICATION REPORT
# ============================================================

with open(
    "models/evaluation/classification_report.txt",
    "w",
) as file:

    file.write(
        f"DocuMind AI — ResNet18 V2\n\n"
    )

    file.write(
        f"Accuracy: "
        f"{accuracy * 100:.2f}%\n\n"
    )

    file.write(
        report
    )


print(
    "models/evaluation/classification_report.txt"
)


# ============================================================
# FINISHED
# ============================================================

print("\n" + "=" * 60)

print("EVALUATION COMPLETE")

print("=" * 60)