from pathlib import Path
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from datasets import load_dataset


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_NAME = "hf-tuner/rvl-cdip-document-classification"

NUM_CLASSES = 16

BATCH_SIZE = 32

EPOCHS = 5

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

MODEL_PATH = "models/document_classifier_v2.pth"

CLASS_NAMES_PATH = "models/class_names.json"


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("DocuMind AI — Model V2 Training")
print("=" * 60)

print(f"Device: {DEVICE}")

if torch.cuda.is_available():

    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )

    print(
        f"CUDA: {torch.version.cuda}"
    )


# ============================================================
# LOAD DATASET
# ============================================================

print("\nLoading dataset...")

dataset = load_dataset(
    DATASET_NAME
)

print(dataset)


# ============================================================
# CLASS INFORMATION
# ============================================================

class_names = dataset[
    "train"
].features[
    "label"
].names

print("\nClasses:")

for index, name in enumerate(
    class_names
):

    print(
        f"{index}: {name}"
    )


# ============================================================
# IMAGE TRANSFORMS
# ============================================================

train_transform = transforms.Compose([

    transforms.Resize(
        (256, 256)
    ),

    transforms.RandomResizedCrop(
        224,
        scale=(0.85, 1.0)
    ),

    transforms.RandomRotation(
        3
    ),

    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15
    ),

    transforms.ToTensor(),

    transforms.Normalize(
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
    ),
])


val_transform = transforms.Compose([

    transforms.Resize(
        (224, 224)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
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
    ),
])


# ============================================================
# DATASET TRANSFORMATION
# ============================================================

def train_transform_image(example):

    example["image"] = [
        train_transform(
            image.convert("RGB")
        )
        for image in example["image"]
    ]

    return example


def val_transform_image(example):

    example["image"] = [
        val_transform(
            image.convert("RGB")
        )
        for image in example["image"]
    ]

    return example


train_dataset = dataset[
    "train"
].with_transform(
    train_transform_image
)


test_dataset = dataset[
    "test"
].with_transform(
    val_transform_image
)


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)


test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)


# ============================================================
# MODEL
# ============================================================

print("\nLoading pretrained ResNet18...")

weights = models.ResNet18_Weights.DEFAULT

model = models.resnet18(
    weights=weights
)


# Replace ImageNet classifier

model.fc = nn.Sequential(

    nn.Dropout(
        p=0.3
    ),

    nn.Linear(
        model.fc.in_features,
        NUM_CLASSES
    )
)


model = model.to(
    DEVICE
)


# ============================================================
# LOSS
# ============================================================

criterion = nn.CrossEntropyLoss(
    label_smoothing=0.1
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# LEARNING RATE SCHEDULER
# ============================================================

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS
)


# ============================================================
# TRAINING
# ============================================================

best_accuracy = 0.0


for epoch in range(EPOCHS):

    print("\n" + "=" * 60)

    print(
        f"Epoch {epoch + 1}/{EPOCHS}"
    )

    print("=" * 60)


    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    running_loss = 0.0

    correct = 0

    total = 0


    for batch in train_loader:

        images = batch[
            "image"
        ].to(
            DEVICE,
            non_blocking=True
        )

        labels = batch[
            "label"
        ].to(
            DEVICE,
            non_blocking=True
        )


        optimizer.zero_grad(
            set_to_none=True
        )


        outputs = model(
            images
        )


        loss = criterion(
            outputs,
            labels
        )


        loss.backward()


        optimizer.step()


        running_loss += (
            loss.item()
        )


        predictions = outputs.argmax(
            dim=1
        )


        total += labels.size(0)


        correct += (
            predictions == labels
        ).sum().item()


    train_loss = (
        running_loss /
        len(train_loader)
    )


    train_accuracy = (
        correct /
        total
    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    val_correct = 0

    val_total = 0

    val_loss = 0.0


    with torch.no_grad():

        for batch in test_loader:

            images = batch[
                "image"
            ].to(
                DEVICE,
                non_blocking=True
            )

            labels = batch[
                "label"
            ].to(
                DEVICE,
                non_blocking=True
            )


            outputs = model(
                images
            )


            loss = criterion(
                outputs,
                labels
            )


            val_loss += (
                loss.item()
            )


            predictions = outputs.argmax(
                dim=1
            )


            val_total += labels.size(0)


            val_correct += (
                predictions == labels
            ).sum().item()


    validation_loss = (
        val_loss /
        len(test_loader)
    )


    validation_accuracy = (
        val_correct /
        val_total
    )


    scheduler.step()


    print(
        f"Train Loss: "
        f"{train_loss:.4f}"
    )

    print(
        f"Train Accuracy: "
        f"{train_accuracy:.4f}"
    )

    print(
        f"Validation Loss: "
        f"{validation_loss:.4f}"
    )

    print(
        f"Validation Accuracy: "
        f"{validation_accuracy:.4f}"
    )

    print(
        f"Learning Rate: "
        f"{scheduler.get_last_lr()[0]:.8f}"
    )


    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if validation_accuracy > best_accuracy:

        best_accuracy = (
            validation_accuracy
        )


        Path(
            "models"
        ).mkdir(
            exist_ok=True
        )


        torch.save(
            model.state_dict(),
            MODEL_PATH
        )


        with open(
            CLASS_NAMES_PATH,
            "w"
        ) as file:

            json.dump(
                class_names,
                file,
                indent=4
            )


        print(
            "\n✓ New best model saved!"
        )


# ============================================================
# FINAL RESULT
# ============================================================

print("\n" + "=" * 60)

print("TRAINING COMPLETE")

print("=" * 60)

print(
    f"Best Validation Accuracy: "
    f"{best_accuracy:.4f}"
)

print(
    f"Best Validation Accuracy: "
    f"{best_accuracy * 100:.2f}%"
)

print(
    f"\nModel saved to:"
    f"\n{MODEL_PATH}"
)