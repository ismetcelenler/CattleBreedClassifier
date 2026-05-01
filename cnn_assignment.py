"""
CNN Hiperparametre Optimizasyonu - Ismet Celenler 202213709071
Balikesir Universitesi - Derin Ogrenme Dersi
"""
import os, json, time, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
np.random.seed(42)
torch.manual_seed(42)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR = "Cattle Breeds"
RESULTS = "results"
IMG_SIZE = 224
BATCH = 32
EPOCHS = 15
NUM_CLS = 5
CLS_NAMES = ['Ayrshire', 'Brown Swiss', 'Holstein Friesian', 'Jersey', 'Red Dane']
os.makedirs(RESULTS, exist_ok=True)

print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ==================== DATA LOADING ====================
print("\n[1] Veri seti yukleniyor...")
images, labels = [], []
for i, cls in enumerate(sorted(os.listdir(DATA_DIR))):
    d = os.path.join(DATA_DIR, cls)
    if not os.path.isdir(d): continue
    for f in os.listdir(d):
        try:
            img = Image.open(os.path.join(d, f)).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
            images.append(np.array(img, dtype=np.float32) / 255.0)
            labels.append(i)
        except: pass

X = np.array(images).transpose(0, 3, 1, 2)  # NCHW
y = np.array(labels)
print(f"  Toplam: {len(X)} goruntu, Siniflar: {dict(zip(CLS_NAMES, np.bincount(y)))}")

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

def make_loader(X, y, shuffle=True):
    ds = TensorDataset(torch.tensor(X), torch.tensor(y, dtype=torch.long))
    return DataLoader(ds, batch_size=BATCH, shuffle=shuffle)

train_loader = make_loader(X_train, y_train)
val_loader = make_loader(X_val, y_val, False)
test_loader = make_loader(X_test, y_test, False)

# Augmentation transform
aug_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
])

# ==================== CONVOLUTION DEMO ====================
print("\n[2] Convolution hesaplama (3x3 kernel, 5x5 input)...")
np.random.seed(26)
inp = np.random.randint(0, 10, (5, 5)).astype(float)
kern = np.random.randint(-2, 3, (3, 3)).astype(float)
out = np.zeros((3, 3))
steps_text = []
for i in range(3):
    for j in range(3):
        region = inp[i:i+3, j:j+3]
        val = np.sum(region * kern)
        out[i, j] = val
        steps_text.append(f"Pozisyon ({i},{j}): sum({region.flatten()} * {kern.flatten()}) = {val:.0f}")

with open(os.path.join(RESULTS, 'conv_steps.txt'), 'w', encoding='utf-8') as f:
    f.write(f"Giris Matrisi (5x5):\n{inp}\n\nKernel (3x3):\n{kern}\n\nCikis (3x3):\n{out}\n\nAdimlar:\n")
    f.write('\n'.join(steps_text))

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
sns.heatmap(inp, annot=True, fmt='.0f', cmap='YlOrRd', ax=axes[0], cbar=False, linewidths=1)
axes[0].set_title('Giris (5x5)')
sns.heatmap(kern, annot=True, fmt='.0f', cmap='RdBu_r', ax=axes[1], cbar=False, linewidths=1)
axes[1].set_title('Kernel (3x3)')
sns.heatmap(out, annot=True, fmt='.0f', cmap='Greens', ax=axes[2], cbar=False, linewidths=1)
axes[2].set_title('Cikis (3x3)')
plt.suptitle('Evrisim (Convolution) Islemi', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS, 'convolution_demo.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Kaydedildi.")

# ==================== MODEL DEFINITIONS ====================
class CustomCNN(nn.Module):
    def __init__(self, dropout=True, l2=False):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 14 * 14, 512), nn.ReLU(),
            nn.Dropout(0.5) if dropout else nn.Identity(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Dropout(0.3) if dropout else nn.Identity(),
            nn.Linear(256, NUM_CLS)
        )
        self.use_l2 = l2
    def forward(self, x): return self.classifier(self.features(x))

def make_transfer(base_cls, name):
    if name == 'VGG16':
        base = base_cls(weights='IMAGENET1K_V1')
        for p in base.features.parameters(): p.requires_grad = False
        base.classifier[-1] = nn.Linear(4096, NUM_CLS)
        return base
    else:
        base = base_cls(weights='IMAGENET1K_V1')
        for p in base.parameters(): p.requires_grad = False
        if hasattr(base, 'fc'):
            in_f = base.fc.in_features
            base.fc = nn.Sequential(nn.Linear(in_f, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, NUM_CLS))
        elif hasattr(base, 'classifier'):
            if isinstance(base.classifier, nn.Sequential):
                in_f = base.classifier[-1].in_features
                base.classifier[-1] = nn.Linear(in_f, NUM_CLS)
            else:
                in_f = base.classifier.in_features
                base.classifier = nn.Sequential(nn.Linear(in_f, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, NUM_CLS))
        return base

# ==================== TRAINING ====================
def train_model(model, name, train_ld, val_ld, epochs=EPOCHS, lr=0.001, use_aug=True):
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_loss = float('inf')
    patience, patience_count = 5, 0
    
    start = time.time()
    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0, 0, 0
        for xb, yb in train_ld:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            if use_aug: xb = aug_transform(xb)
            out = model(xb)
            loss = criterion(out, yb)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            running_loss += loss.item() * xb.size(0)
            correct += (out.argmax(1) == yb).sum().item()
            total += yb.size(0)
        train_loss = running_loss / total
        train_acc = correct / total
        
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for xb, yb in val_ld:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                out = model(xb)
                val_loss += criterion(out, yb).item() * xb.size(0)
                val_correct += (out.argmax(1) == yb).sum().item()
                val_total += yb.size(0)
        val_loss /= val_total
        val_acc = val_correct / val_total
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        scheduler.step(val_loss)
        
        print(f"  Epoch {epoch+1}/{epochs} - loss: {train_loss:.4f} acc: {train_acc:.4f} val_loss: {val_loss:.4f} val_acc: {val_acc:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_count = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break
    
    model.load_state_dict(best_state)
    elapsed = time.time() - start
    print(f"  {name} sure: {elapsed:.1f}s")
    return model, history, elapsed

def evaluate(model, loader):
    model.eval()
    all_pred, all_true, all_probs = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            out = model(xb)
            probs = torch.softmax(out, dim=1)
            all_pred.extend(out.argmax(1).cpu().numpy())
            all_true.extend(yb.numpy())
            all_probs.extend(probs.cpu().numpy())
    return np.array(all_true), np.array(all_pred), np.array(all_probs)

def plot_history(h, name, path):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    a1.plot(h['train_acc'], label='Train'); a1.plot(h['val_acc'], label='Val')
    a1.set_title(f'{name} - Accuracy'); a1.legend(); a1.grid(True, alpha=0.3)
    a2.plot(h['train_loss'], label='Train'); a2.plot(h['val_loss'], label='Val')
    a2.set_title(f'{name} - Loss'); a2.legend(); a2.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()

def plot_cm(cm, name, path):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLS_NAMES, yticklabels=CLS_NAMES, ax=ax)
    ax.set_title(f'{name} - Confusion Matrix'); ax.set_xlabel('Tahmin'); ax.set_ylabel('Gercek')
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()

# ==================== TRAIN 5 MODELS ====================
print("\n[3] 5 model egitiliyor...")
results = {}

model_configs = [
    ('Custom_CNN', lambda: CustomCNN(dropout=True)),
    ('VGG16', lambda: make_transfer(models.vgg16, 'VGG16')),
    ('ResNet50', lambda: make_transfer(models.resnet50, 'ResNet50')),
    ('MobileNetV2', lambda: make_transfer(models.mobilenet_v2, 'MobileNetV2')),
    ('EfficientNetB0', lambda: make_transfer(models.efficientnet_b0, 'EfficientNetB0')),
]

for name, build_fn in model_configs:
    print(f"\n--- {name} ---")
    model = build_fn()
    param_total = sum(p.numel() for p in model.parameters())
    param_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Params: {param_total:,} total, {param_train:,} trainable")
    
    model, hist, elapsed = train_model(model, name, train_loader, val_loader)
    y_true, y_pred, y_probs = evaluate(model, test_loader)
    
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=CLS_NAMES, output_dict=True)
    acc = (y_true == y_pred).mean()
    
    results[name] = {
        'accuracy': float(acc), 'params_total': param_total, 'params_train': param_train,
        'time': elapsed, 'history': hist, 'cm': cm, 'report': report,
        'y_true': y_true, 'y_pred': y_pred, 'y_probs': y_probs, 'model': model
    }
    
    plot_history(hist, name, os.path.join(RESULTS, f'history_{name}.png'))
    plot_cm(cm, name, os.path.join(RESULTS, f'cm_{name}.png'))
    print(f"  {name} Test Accuracy: {acc:.4f}")
    
    if name not in ['Custom_CNN']:
        del model
        torch.cuda.empty_cache()

# ==================== SCENARIO 1: OVERFITTING ====================
print("\n[4] Senaryo 1: Overfitting Deneyi...")
print("--- No Dropout/Reg ---")
m_nodrp = CustomCNN(dropout=False)
m_nodrp, h_nodrp, _ = train_model(m_nodrp, 'CNN_NoDrop', train_loader, val_loader, epochs=20, use_aug=False)

print("--- With Dropout + L2 ---")
m_drp = CustomCNN(dropout=True, l2=True)
wd_optimizer = True
m_drp = m_drp.to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer_l2 = optim.Adam(m_drp.parameters(), lr=0.001, weight_decay=1e-4)
h_drp = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
for epoch in range(20):
    m_drp.train()
    rl, c, t = 0, 0, 0
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        out = m_drp(xb); loss = criterion(out, yb)
        optimizer_l2.zero_grad(); loss.backward(); optimizer_l2.step()
        rl += loss.item()*xb.size(0); c += (out.argmax(1)==yb).sum().item(); t += yb.size(0)
    m_drp.eval()
    vl, vc, vt = 0, 0, 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            out = m_drp(xb); vl += criterion(out,yb).item()*xb.size(0)
            vc += (out.argmax(1)==yb).sum().item(); vt += yb.size(0)
    h_drp['train_loss'].append(rl/t); h_drp['val_loss'].append(vl/vt)
    h_drp['train_acc'].append(c/t); h_drp['val_acc'].append(vc/vt)
    print(f"  Epoch {epoch+1}/20 - loss:{rl/t:.4f} acc:{c/t:.4f} val_loss:{vl/vt:.4f} val_acc:{vc/vt:.4f}")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
a1.plot(h_nodrp['train_acc'], 'r-', label='Train (No Reg)'); a1.plot(h_nodrp['val_acc'], 'r--', label='Val (No Reg)')
a1.plot(h_drp['train_acc'], 'b-', label='Train (Dropout+L2)'); a1.plot(h_drp['val_acc'], 'b--', label='Val (Dropout+L2)')
a1.set_title('Overfitting Deneyi - Accuracy'); a1.legend(); a1.grid(True, alpha=0.3)
a2.plot(h_nodrp['train_loss'], 'r-', label='Train (No Reg)'); a2.plot(h_nodrp['val_loss'], 'r--', label='Val (No Reg)')
a2.plot(h_drp['train_loss'], 'b-', label='Train (Dropout+L2)'); a2.plot(h_drp['val_loss'], 'b--', label='Val (Dropout+L2)')
a2.set_title('Overfitting Deneyi - Loss'); a2.legend(); a2.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(RESULTS, 'scenario1_overfitting.png'), dpi=150, bbox_inches='tight'); plt.close()
del m_nodrp, m_drp; torch.cuda.empty_cache()

# ==================== SCENARIO 2: LEARNING RATE ====================
print("\n[5] Senaryo 2: Learning Rate karsilastirma...")
lr_histories = {}
for lr in [0.1, 0.001, 0.0001]:
    print(f"\n--- LR = {lr} ---")
    m = CustomCNN(dropout=True)
    m, h, _ = train_model(m, f'CNN_LR{lr}', train_loader, val_loader, epochs=15, lr=lr)
    lr_histories[lr] = h
    del m; torch.cuda.empty_cache()

fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
colors = {'0.1': 'r', '0.001': 'g', '0.0001': 'b'}
for lr, h in lr_histories.items():
    c = colors.get(str(lr), 'k')
    a1.plot(h['val_acc'], color=c, label=f'LR={lr}')
    a2.plot(h['val_loss'], color=c, label=f'LR={lr}')
a1.set_title('Learning Rate - Val Accuracy'); a1.legend(); a1.grid(True, alpha=0.3)
a2.set_title('Learning Rate - Val Loss'); a2.legend(); a2.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(RESULTS, 'scenario2_lr.png'), dpi=150, bbox_inches='tight'); plt.close()

# ==================== SCENARIO 3: TRANSFER LEARNING ====================
print("\n[6] Senaryo 3: Transfer Learning vs Scratch...")
print("--- ResNet50 Fine-tuning ---")
m_ft = make_transfer(models.resnet50, 'ResNet50')
m_ft, h_ft, _ = train_model(m_ft, 'ResNet50_FT', train_loader, val_loader)

print("--- ResNet50 From Scratch ---")
m_sc = models.resnet50(weights=None, num_classes=NUM_CLS)
m_sc, h_sc, _ = train_model(m_sc, 'ResNet50_Scratch', train_loader, val_loader)

fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
a1.plot(h_ft['val_acc'], 'b-', label='Fine-tuning'); a1.plot(h_sc['val_acc'], 'r-', label='From Scratch')
a1.set_title('Transfer Learning - Val Accuracy'); a1.legend(); a1.grid(True, alpha=0.3)
a2.plot(h_ft['val_loss'], 'b-', label='Fine-tuning'); a2.plot(h_sc['val_loss'], 'r-', label='From Scratch')
a2.set_title('Transfer Learning - Val Loss'); a2.legend(); a2.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(RESULTS, 'scenario3_transfer.png'), dpi=150, bbox_inches='tight'); plt.close()
del m_ft, m_sc; torch.cuda.empty_cache()

# ==================== COMPARISON ====================
print("\n[7] Model karsilastirma...")
fig, ax = plt.subplots(figsize=(10, 6))
names = list(results.keys())
accs = [results[n]['accuracy'] for n in names]
bars = ax.bar(names, accs, color=sns.color_palette('viridis', len(names)))
for bar, acc in zip(bars, accs):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005, f'{acc:.2%}', ha='center', fontweight='bold')
ax.set_title('Model Karsilastirma - Test Accuracy'); ax.set_ylim(0, 1.05); ax.grid(axis='y', alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(RESULTS, 'model_comparison.png'), dpi=150, bbox_inches='tight'); plt.close()

# Param comparison table
print("\n  Model Parametre Karsilastirmasi:")
print(f"  {'Model':<20} {'Total Params':>15} {'Trainable':>15} {'Accuracy':>10} {'Sure(s)':>10}")
for n in names:
    r = results[n]
    print(f"  {n:<20} {r['params_total']:>15,} {r['params_train']:>15,} {r['accuracy']:>10.4f} {r['time']:>10.1f}")

# Save param info
param_data = {n: {'total': results[n]['params_total'], 'trainable': results[n]['params_train'],
                  'accuracy': results[n]['accuracy'], 'time': results[n]['time']} for n in names}
with open(os.path.join(RESULTS, 'param_comparison.json'), 'w') as f:
    json.dump(param_data, f, indent=2)

# ==================== ERROR ANALYSIS ====================
print("\n[8] Hata analizi...")
best_name = max(results, key=lambda n: results[n]['accuracy'])
best_r = results[best_name]
wrong_idx = np.where(best_r['y_true'] != best_r['y_pred'])[0]

n_show = min(10, len(wrong_idx))
if n_show > 0:
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    for idx in range(n_show):
        i = wrong_idx[idx]
        # Get the test image
        img = X_test[i].transpose(1, 2, 0)  # CHW -> HWC
        axes[idx].imshow(img)
        true_cls = CLS_NAMES[best_r['y_true'][i]]
        pred_cls = CLS_NAMES[best_r['y_pred'][i]]
        conf = best_r['y_probs'][i][best_r['y_pred'][i]]
        axes[idx].set_title(f'Gercek: {true_cls}\nTahmin: {pred_cls}\nGuven: {conf:.2f}', fontsize=9)
        axes[idx].axis('off')
    for idx in range(n_show, 10):
        axes[idx].axis('off')
    plt.suptitle(f'Hata Analizi - {best_name} Yanlis Tahminleri', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, 'error_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  {len(wrong_idx)} yanlis tahmin, {n_show} tanesi gorsellendi.")

# ==================== SAVE SUMMARY ====================
print("\n[9] Sonuclar kaydediliyor...")
summary = {}
for n in names:
    r = results[n]
    summary[n] = {
        'accuracy': r['accuracy'],
        'params_total': r['params_total'],
        'params_trainable': r['params_train'],
        'training_time': r['time'],
        'precision_macro': r['report']['macro avg']['precision'],
        'recall_macro': r['report']['macro avg']['recall'],
        'f1_macro': r['report']['macro avg']['f1-score'],
        'per_class': {c: {'precision': r['report'][c]['precision'],
                          'recall': r['report'][c]['recall'],
                          'f1': r['report'][c]['f1-score']} for c in CLS_NAMES}
    }
with open(os.path.join(RESULTS, 'summary.json'), 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print("\n" + "="*60)
print("TAMAMLANDI!")
print("="*60)
for n in names:
    print(f"  {n}: {results[n]['accuracy']:.2%}")
print(f"\nEn iyi model: {best_name} ({results[best_name]['accuracy']:.2%})")
print(f"Sonuclar '{RESULTS}/' klasorune kaydedildi.")
