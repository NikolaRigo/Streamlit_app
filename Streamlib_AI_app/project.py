import os
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from tqdm import tqdm
import kagglehub

path = kagglehub.dataset_download("aliabdelmenam/rdd-2022")
print("Dataset path:", path)

LABEL_MAP = {
    "D00": 0,
    "D10": 1,
    "D20": 2,
    "D40": 3
}
NUM_CLASSES = 4
CLASS_NAMES = ["Longitudinal Crack","Transverse Crack","Alligator Crack","Pothole"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 4
EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
DATASET_ROOT = path
MODEL_SAVE_PATH = "best_road_model.pth"# do NOT add /train or /images
print(os.listdir(DATASET_ROOT))

VALID_CLASSES = {0, 1, 2, 3}
class RDDDataset(Dataset):
    def __init__(self, root, split="train"):
        self.img_dir = os.path.join(root, "RDD_SPLIT", split, "images")
        self.lbl_dir = os.path.join(root, "RDD_SPLIT", split, "labels")

        self.images = [f for f in os.listdir(self.img_dir) if f.endswith(".jpg")]

        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]

        img_path = os.path.join(self.img_dir, img_name)
        lbl_path = os.path.join(self.lbl_dir, img_name.replace(".jpg", ".txt"))

        image = Image.open(img_path).convert("RGB")

        labels = []
        if os.path.exists(lbl_path):
            with open(lbl_path) as f:
                for line in f:
                    cls_id = int(line.split()[0])
                    labels.append(cls_id)

        # Use most severe label (pothole > cracks)
        label = max(labels) if labels else 0
        label = min(label, 3)  # force labels into 0–3

        return self.transform(image), label

train_ds = RDDDataset(DATASET_ROOT, split="train")
val_ds   = RDDDataset(DATASET_ROOT, split="val")

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

ys = [train_ds[i][1] for i in range(500)]
print(sorted(set(ys)))

'''device = "cuda" if torch.cuda.is_available() else "cpu"

model = models.mobilenet_v2(pretrained=True)
model.classifier[1] = nn.Linear(model.last_channel, NUM_CLASSES)
model.to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

model.train()

for i, (x, y) in enumerate(train_loader):
    x, y = x.to(device), y.to(device)

    optimizer.zero_grad()
    out = model(x)
    loss = loss_fn(out, y)
    loss.backward()
    optimizer.step()

    print(f"Batch {i}, Loss: {loss.item():.4f}")

    if i == 10:
        break
'''

model = models.mobilenet_v2(pretrained=True)
model.classifier[1] = nn.Linear(model.last_channel, NUM_CLASSES)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

best_val_acc = 0.0

for epoch in range(EPOCHS):
    # ----- Train -----
    model.train()

    loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}", leave=True)
    running_loss = 0
    correct = 0
    total = 0

    for imgs, labels in loop:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        loop.set_postfix(loss=loss.item(), acc=correct/total)

    train_loss = running_loss / total
    train_acc = correct / total

    # ----- Validation -----
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            _, preds = outputs.max(1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_acc = val_correct / val_total
    print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    # ----- Storing Functionality -----
    # Save the model only if validation accuracy improves
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"⭐ New Best Model Saved! Accuracy: {val_acc:.4f}")

print(f"\nTraining Complete. Best Model is stored at: {os.path.abspath(MODEL_SAVE_PATH)}")