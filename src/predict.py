import json

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


MODEL_PATH = "models/document_classifier_v2.pth"
CLASS_NAMES_PATH = "models/class_names.json"


class DocumentPredictor:

    def __init__(
        self,
        model_path=MODEL_PATH,
        classes_path=CLASS_NAMES_PATH,
    ):

        # ----------------------------------------
        # Device
        # ----------------------------------------

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        # ----------------------------------------
        # Load class names
        # ----------------------------------------

        with open(
            classes_path,
            "r",
        ) as file:

            self.class_names = json.load(
                file
            )

        # ----------------------------------------
        # Create V2 ResNet18 architecture
        # ----------------------------------------

        self.model = models.resnet18(
            weights=None
        )

        # IMPORTANT:
        # V2 was trained with:
        #
        # Dropout → Linear
        #
        # So inference must use the same architecture.

        self.model.fc = nn.Sequential(

            nn.Dropout(
                p=0.3
            ),

            nn.Linear(
                self.model.fc.in_features,
                len(self.class_names),
            ),
        )

        # ----------------------------------------
        # Load V2 weights
        # ----------------------------------------

        self.model.load_state_dict(
            torch.load(
                model_path,
                map_location=self.device,
                weights_only=True,
            )
        )

        # ----------------------------------------
        # Move model to GPU / CPU
        # ----------------------------------------

        self.model = self.model.to(
            self.device
        )

        self.model.eval()

        # ----------------------------------------
        # Image preprocessing
        # ----------------------------------------

        self.transform = transforms.Compose([

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

    # ====================================================
    # PREDICTION
    # ====================================================

    def predict(
        self,
        image_path,
    ):

        # ----------------------------------------
        # Load image
        # ----------------------------------------

        image = Image.open(
            image_path
        ).convert("RGB")

        # ----------------------------------------
        # Transform image
        # ----------------------------------------

        image_tensor = self.transform(
            image
        ).unsqueeze(0)

        image_tensor = image_tensor.to(
            self.device
        )

        # ----------------------------------------
        # Model inference
        # ----------------------------------------

        with torch.no_grad():

            outputs = self.model(
                image_tensor
            )

            probabilities = torch.softmax(
                outputs,
                dim=1,
            )

            # Get top 3 predictions

            top_probabilities, top_indices = torch.topk(
                probabilities,
                k=3,
                dim=1,
            )

        # ----------------------------------------
        # Format predictions
        # ----------------------------------------

        predictions = []

        for probability, index in zip(
            top_probabilities[0],
            top_indices[0],
        ):

            predictions.append({

                "document_type":
                    self.class_names[
                        index.item()
                    ],

                "confidence":
                    round(
                        probability.item()
                        * 100,
                        2,
                    ),
            })

        # ----------------------------------------
        # Return result
        # ----------------------------------------

        return {

            "document_type":
                predictions[0][
                    "document_type"
                ],

            "confidence":
                predictions[0][
                    "confidence"
                ],

            "top_predictions":
                predictions,

            "device":
                str(self.device),
        }