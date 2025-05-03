from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader, Subset

import torch.nn as nn
import torch.nn.functional as F

import torch

transform = transforms.Compose([
    transforms.Grayscale(),  # ensure single channel
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Replace EMNIST dataset loading with ImageFolder
train_data = ImageFolder(root='./dataset_root/train', transform=transform)
test_data = ImageFolder(root='./dataset_root/test', transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)


# basic CNN
class LetterClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 26)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # (32x32 → 16x16)
        x = self.pool(F.relu(self.conv2(x)))   # (16x16 → 8x8)
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# training the model using CrossEntropyLoss
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LetterClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for images, labels in train_loader:
    print("Batch labels:", labels)
    break

for epoch in range(5):
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), (labels).to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")

# loop through testloader and evaluate
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), (labels).to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

print(f"Test Accuracy: {100 * correct / total:.2f}%")

torch.save(model.state_dict(), 'letter_classifier.pth')
print("Model saved to letter_classifier.pth")


