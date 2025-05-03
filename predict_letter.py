import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms
from PIL import Image, ImageOps

import matplotlib.pyplot as plt

# 1. Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

model = LetterClassifier().to(device)

# 3. Load saved weights
model.load_state_dict(torch.load('letter_classifier.pth', map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

image_path = './dataset_root/test/J/J_34.png'  # change this to your file path
img = Image.open(image_path).convert('L')  # Convert to grayscale

plt.imshow(img, cmap='gray')
plt.title("Model Input")
plt.show()

img = transform(img)           # shape: (1, 32, 32)
img = img.unsqueeze(0).to(device)  # add batch dimension → (1, 1, 32, 32)

with torch.no_grad():
    output = model(img)  # shape: (1, 26)
    _, predicted = torch.max(output, 1)
    print(output)

# 7. Map number to letter
classes = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
predicted_letter = classes[predicted.item()]

print(f"Predicted letter: {predicted_letter}")
