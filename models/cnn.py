
import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, num_classes=10, input_size=32):
        """
        CNN model for image classification.
        
        Args:
            num_classes : number of output classes (default 10)
            input_size  : image size — 32 for CIFAR-10, 64 for Tiny ImageNet
        """
        super(CNN, self).__init__()

        # ── Block 1 ──────────────────────────────────────────
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25)
        )

        # ── Block 2 ──────────────────────────────────────────
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25)
        )

        # ── Classifier ───────────────────────────────────────
        flat_size = 128 * (input_size // 4) * (input_size // 4)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.classifier(x)
        return x
