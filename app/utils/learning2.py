from tqdm import tqdm
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from PIL import Image
from torchvision.datasets.folder import default_loader
def safe_loader(path):
    try:
        return default_loader(path)
    except OSError:
        print(f"⚠️ Skipping corrupted image: {path}")
        return Image.new('RGB', (224, 224))
if __name__ == "__main__":
    DATASET_DIR = '../../dataset_styles'
    BATCH_SIZE = 64
    EPOCHS = 25
    LR = 1e-4
    IMG_SIZE = 224
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.RandomPerspective(distortion_scale=0.3, p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    train_data = ImageFolder(root=os.path.join(DATASET_DIR, 'train'), transform=train_transform, loader=safe_loader)
    val_data = ImageFolder(root=os.path.join(DATASET_DIR, 'val'), transform=val_transform, loader=safe_loader)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=os.cpu_count(), pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, num_workers=os.cpu_count(), pin_memory=True)
    class_names = train_data.classes
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model = model.to(device)
    for param in model.parameters():
        param.requires_grad = True
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)  # снижать lr каждые 4 эпохи
    best_acc = 0.0
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Training]"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.detach()
        train_loss = running_loss / len(train_loader)
        print(f"\nEpoch {epoch+1} - Train Loss: {train_loss:.4f}")
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Validation]"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        acc = correct / total
        val_loss /= len(val_loader)
        print(f"Validation Accuracy: {acc:.4f} | Val Loss: {val_loss:.4f}")
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), '../../models/model_styles.pth')
            print(f"New best model saved (epoch {epoch+1}, acc {acc:.4f})")
    print("Training complete. Best Accuracy:", best_acc)
