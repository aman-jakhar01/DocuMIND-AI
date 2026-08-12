from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
from PIL import Image


class DocumentClassifier:
    def __init__(self, num_classes: int):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        weights = models.ResNet18_Weights.DEFAULT

        self.model = models.resnet18(weights=weights)

        # Replace ImageNet's 1000-class output layer
        self.model.fc = nn.Linear(
            self.model.fc.in_features,
            num_classes
        )

        self.model = self.model.to(self.device)

        self.train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(5),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        self.val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def train(
        self,
        train_loader,
        val_loader,
        epochs=5,
        learning_rate=1e-4,
    ):
        criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate
        )

        for epoch in range(epochs):
            self.model.train()

            running_loss = 0.0
            correct = 0
            total = 0

            for images, labels in train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()

                outputs = self.model(images)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                _, predicted = torch.max(outputs, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            train_accuracy = correct / total

            val_accuracy = self.evaluate(val_loader)

            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"Loss: {running_loss / len(train_loader):.4f} | "
                f"Train Accuracy: {train_accuracy:.4f} | "
                f"Validation Accuracy: {val_accuracy:.4f}"
            )

    def evaluate(self, loader):
        self.model.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)

                _, predicted = torch.max(outputs, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return correct / total

    def save(self, path="models/document_classifier.pth"):
        Path(path).parent.mkdir(
            parents=True,
            exist_ok=True
        )

        torch.save(
            self.model.state_dict(),
            path
        )

        print(f"Model saved to {path}")