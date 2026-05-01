"""EfficientNetB0 modelini kaydet"""
import os, numpy as np, torch
from torch import nn
from torchvision import models
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms

torch.manual_seed(42); np.random.seed(42)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = 224; NUM_CLS = 5; BATCH = 32; EPOCHS = 20
CLS_NAMES = ['Ayrshire', 'Brown Swiss', 'Holstein Friesian', 'Jersey', 'Red Dane']

print(f"Device: {DEVICE}")

# Load data
images, labels = [], []
for i, cls in enumerate(sorted(os.listdir("Cattle Breeds"))):
    d = os.path.join("Cattle Breeds", cls)
    if not os.path.isdir(d): continue
    for f in os.listdir(d):
        try:
            img = Image.open(os.path.join(d, f)).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
            images.append(np.array(img, dtype=np.float32) / 255.0)
            labels.append(i)
        except: pass

X = np.array(images).transpose(0, 3, 1, 2)
y = np.array(labels)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train, dtype=torch.long))
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
aug = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomRotation(15), transforms.ColorJitter(0.2, 0.2)])

# Build model
base = models.efficientnet_b0(weights='IMAGENET1K_V1')
for p in base.parameters(): p.requires_grad = False
in_f = base.classifier[-1].in_features
base.classifier[-1] = nn.Linear(in_f, NUM_CLS)
model = base.to(DEVICE)

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
criterion = nn.CrossEntropyLoss()

print("Egitim basliyor...")
for epoch in range(EPOCHS):
    model.train(); correct = total = 0
    for xb, yb in train_loader:
        xb, yb = aug(xb.to(DEVICE)), yb.to(DEVICE)
        out = model(xb); loss = criterion(out, yb)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        correct += (out.argmax(1)==yb).sum().item(); total += yb.size(0)
    print(f"  Epoch {epoch+1}/{EPOCHS} - acc: {correct/total:.4f}")

torch.save({'model_state': model.state_dict(), 'class_names': CLS_NAMES}, 'efficientnet_cattle.pth')
print("Model kaydedildi: efficientnet_cattle.pth")
