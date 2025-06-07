import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from tqdm import tqdm
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-4
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])
train_genre = ImageFolder('../../dataset_genre/train', transform=transform)
val_genre = ImageFolder('../../dataset_genre/val', transform=transform)
train_style = ImageFolder('../../dataset_styles/train', transform=transform)
val_style = ImageFolder('../../dataset_styles/val', transform=transform)
train_loader = DataLoader(list(zip(train_genre.samples, train_style.samples)), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(list(zip(val_genre.samples, val_style.samples)), batch_size=BATCH_SIZE)
num_genres = len(train_genre.classes)
num_styles = len(train_style.classes)
class DualHeadResNet(nn.Module):
    def __init__(self, num_genres, num_styles):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        self.genre_fc = nn.Linear(base.fc.in_features, num_genres)
        self.style_fc = nn.Linear(base.fc.in_features, num_styles)
    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        genre_logits = self.genre_fc(x)
        style_logits = self.style_fc(x)
        return genre_logits, style_logits
model = DualHeadResNet(num_genres, num_styles).to(DEVICE)
criterion_genre = nn.CrossEntropyLoss()
criterion_style = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0
    correct_genre = 0
    correct_style = 0
    total = 0
    for (genre_sample, style_sample) in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Training]"):
        genre_paths, genre_labels = zip(*genre_sample)
        style_paths, style_labels = zip(*style_sample)
        assert genre_paths == style_paths, "Mismatch in image paths"
        inputs = torch.stack([transform(ImageFolder.loader(train_genre, p)) for p in genre_paths]).to(DEVICE)
        genre_labels = torch.tensor(genre_labels).to(DEVICE)
        style_labels = torch.tensor(style_labels).to(DEVICE)
        optimizer.zero_grad()
        genre_out, style_out = model(inputs)
        loss_genre = criterion_genre(genre_out, genre_labels)
        loss_style = criterion_style(style_out, style_labels)
        loss = loss_genre + loss_style
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        correct_genre += (genre_out.argmax(1) == genre_labels).sum().item()
        correct_style += (style_out.argmax(1) == style_labels).sum().item()
        total += inputs.size(0)
    print(f"Epoch {epoch+1}: Loss={running_loss:.4f}, Genre Acc={correct_genre/total:.4f}, Style Acc={correct_style/total:.4f}")
torch.save(model.state_dict(), 'dual_model.pth')
print("✅ Model saved as dual_model.pth")
