# обучение модели на 2 датасетах
from tqdm import tqdm
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from itertools import zip_longest
from tqdm import tqdm

DATASET_GENRE = 'dataset_genre'  # жанры
DATASET_STYLE = 'dataset_styles'  # стили
BATCH_SIZE = 64
EPOCHS = 5
LR = 1e-4
IMG_SIZE = 224
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])
def rgb_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
genre_train = ImageFolder(root=os.path.join(DATASET_GENRE, 'train'), transform=train_transform, loader=rgb_loader)
genre_val = ImageFolder(root=os.path.join(DATASET_GENRE, 'val'), transform=val_transform, loader=rgb_loader)
style_train = ImageFolder(root=os.path.join(DATASET_STYLE, 'train'), transform=train_transform, loader=rgb_loader)
style_val = ImageFolder(root=os.path.join(DATASET_STYLE, 'val'), transform=val_transform, loader=rgb_loader)
genre_loader = DataLoader(genre_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
genre_val_loader = DataLoader(genre_val, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)
style_loader = DataLoader(style_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
style_val_loader = DataLoader(style_val, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)
class DualClassifier(nn.Module):
    def __init__(self, base_model, n_genres, n_styles):
        super().__init__()
        self.backbone = nn.Sequential(*list(base_model.children())[:-1])
        self.genre_head = nn.Linear(base_model.fc.in_features, n_genres)
        self.style_head = nn.Linear(base_model.fc.in_features, n_styles)

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        return self.genre_head(x), self.style_head(x)
base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model = DualClassifier(base, len(genre_train.classes), len(style_train.classes))
model = model.to(device)
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
os.makedirs('../../models', exist_ok=True)
def train():
    print("Using device:", device)
    print("Model is on:", next(model.parameters()).device)
    best_acc = 0.0
    for epoch in range(EPOCHS):
        model.train()
        genre_loss_total = 0.0
        style_loss_total = 0.0
        progress_bar = tqdm(zip_longest(genre_loader, style_loader), total=min(len(genre_loader), len(style_loader)),
                            desc=f"Epoch {epoch + 1}/{EPOCHS} [Training]", leave=True, dynamic_ncols=True)
        for genre_batch, style_batch in progress_bar:
            if genre_batch is None or style_batch is None:
                continue
            (g_inputs, g_labels) = genre_batch
            (s_inputs, s_labels) = style_batch
            inputs = torch.cat((g_inputs, s_inputs), 0).to(device)
            genre_targets = g_labels.to(device)
            style_targets = s_labels.to(device)
            optimizer.zero_grad()
            genre_out, style_out = model(inputs)
            loss_genre = criterion(genre_out[:len(genre_targets)], genre_targets)
            loss_style = criterion(style_out[len(genre_targets):], style_targets)
            loss = loss_genre + loss_style
            loss.backward()
            optimizer.step()
            genre_loss_total += loss_genre.item()
            style_loss_total += loss_style.item()
        print(f"\nEpoch {epoch+1} - Genre Loss: {genre_loss_total/len(genre_loader):.4f} | Style Loss: {style_loss_total/len(style_loader):.4f}")
        model.eval()
        genre_correct = 0
        style_correct = 0
        genre_total = 0
        style_total = 0
        with torch.no_grad():
            for inputs, labels in tqdm(genre_val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Validation Genre]"):
                inputs, labels = inputs.to(device), labels.to(device)
                genre_out, _ = model(inputs)
                _, preds = torch.max(genre_out, 1)
                genre_correct += (preds == labels).sum().item()
                genre_total += labels.size(0)
            for inputs, labels in tqdm(style_val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Validation Style]"):
                inputs, labels = inputs.to(device), labels.to(device)
                _, style_out = model(inputs)
                _, preds = torch.max(style_out, 1)
                style_correct += (preds == labels).sum().item()
                style_total += labels.size(0)
        genre_acc = genre_correct / genre_total
        style_acc = style_correct / style_total
        print(f"Genre Accuracy: {genre_acc:.4f} | Style Accuracy: {style_acc:.4f}")
        if genre_acc + style_acc > best_acc:
            best_acc = genre_acc + style_acc
            torch.save(model.state_dict(), 'models/model_dual.pth') # путь
            print(f"New best model saved (epoch {epoch+1})")
    print("Training complete. Best Combined Accuracy:", best_acc)
if __name__ == '__main__':
    train()