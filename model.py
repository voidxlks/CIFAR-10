import torch
import torch.nn as nn
import torch.nn.functional as F

class Cifar10CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(Cifar10CNN, self).__init__()

        # Conv layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)

        # Fully connected
        self.fc1 = nn.Linear(128 * 4 * 4, 256)  # correct for 32x32 input
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # 32x32 -> 16x16
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # 16x16 -> 8x8
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # 8x8 -> 4x4
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        # Flatten
        x = x.view(x.size(0), -1)

        # FC layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x
