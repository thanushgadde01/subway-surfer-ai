"""
Subway Surfers CNN training pipeline — Phase 2: Frame Stacking.

Changes from Phase 1:
- SubwayDataset now loads stacks of 4 consecutive frames [4 × 96 × 96]
  instead of single frames [1 × 96 × 96].
- Frames within each class folder are sorted chronologically by filename
  (timestamps), so that frame i and its 3 predecessors form a meaningful
  temporal window.
- When i < 3 (start of a recording session), the earliest available frame
  is repeated as padding.
- Horizontal-flip augmentation is applied identically across all 4 frames
  in a stack so the temporal ordering is preserved.
- SubwayCNN now accepts in_channels=4.
- Checkpoint metadata updated to in_channels=4.
"""

# %%
import cv2
import os
from PIL import Image
import matplotlib.pyplot as plt

from config import game_region

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Dataset
import torch
import torch.nn as nn
import time
import random
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
SAVE_PATH   = os.path.join(BASE_DIR, "models", "subway_cnn.pth")
os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)

ACTIONS = {
    "roll":  0,
    "left":  1,
    "noop":  2,
    "right": 3,
    "jump":  4,
}

STACK_SIZE  = 4      # number of consecutive frames to stack
BATCH_SIZE  = 32
EPOCHS      = 25
LR          = 1e-3
VAL_SPLIT   = 0.15
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

# %%
# ──────────────────────────────────────────────────────────────────────────────
# MODEL ARCHITECTURE  (in_channels = STACK_SIZE)
# ──────────────────────────────────────────────────────────────────────────────
class SubwayCNN(nn.Module):
    """
    DeepMind-style CNN.
    Input:  [batch, STACK_SIZE, 96, 96]
    Output: [batch, num_actions]
    """
    def __init__(self, in_channels=STACK_SIZE, num_actions=5):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),  # → 23×23
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),           # → 10×10
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),           # → 8×8
            nn.ReLU(),
            nn.Flatten(),                                          # → 4096
            nn.Linear(4096, 512),
            nn.ReLU(),
        )
        self.head = nn.Linear(512, num_actions)

    def forward(self, x):
        return self.head(self.backbone(x))


# %%
# ──────────────────────────────────────────────────────────────────────────────
# PREPROCESSING TRANSFORMS
# ──────────────────────────────────────────────────────────────────────────────
class CropRegionTransform:
    """Crop the game window from a full-screen PIL image."""
    def __call__(self, img):
        left   = game_region["left"]
        top    = game_region["top"]
        right  = left + game_region["width"]
        bottom = top  + game_region["height"]
        return img.crop((left, top, right, bottom))


# Single-frame transform: crop → grayscale → 96×96 → tensor [1, 96, 96]
frame_transform = transforms.Compose([
    CropRegionTransform(),
    transforms.Grayscale(),
    transforms.Resize((96, 96)),
    transforms.ToTensor(),          # → [1, 96, 96]
])

# Alias used by play.py for screen-capture preprocessing
transform = frame_transform


# %%
# ──────────────────────────────────────────────────────────────────────────────
# DATASET WITH FRAME STACKING
# ──────────────────────────────────────────────────────────────────────────────
class SubwayDataset(Dataset):
    """
    Loads groups of STACK_SIZE consecutive frames per class folder.

    Each item returned is:
        stacked_tensor : [STACK_SIZE, 96, 96]   (float32 in [0,1])
        label          : int
    """

    def __init__(self, dataset_dir=DATASET_DIR, transform=None, stack_size=STACK_SIZE):
        self.dataset_dir = dataset_dir
        self.transform   = transform
        self.stack_size  = stack_size
        self.flip_map    = {1: 3, 3: 1}   # left ↔ right

        # Build per-class sorted file lists
        # samples: list of (class_paths_sorted, index_within_class, label, to_flip)
        self.samples = []

        for class_name, label in ACTIONS.items():
            folder = os.path.join(dataset_dir, class_name)
            if not os.path.isdir(folder):
                continue

            # Sort filenames so timestamps are chronological
            files = sorted(
                [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".png"))],
            )
            paths = [os.path.join(folder, f) for f in files]

            for i in range(len(paths)):
                # Original
                self.samples.append((paths, i, label, False))
                # Horizontally flipped copy
                self.samples.append((paths, i, label, True))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        paths, i, label, to_flip = self.samples[idx]

        # Build a stack of STACK_SIZE consecutive frames, padding at start
        frames = []
        for offset in range(-(self.stack_size - 1), 1):          # -3, -2, -1, 0
            frame_idx = max(0, i + offset)                         # clamp to 0
            img = Image.open(paths[frame_idx]).convert("RGB")

            if to_flip:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)

            if self.transform:
                img = self.transform(img)    # → tensor [1, 96, 96]

            frames.append(img)              # list of [1, 96, 96]

        stacked = torch.cat(frames, dim=0)  # → [STACK_SIZE, 96, 96]

        if to_flip:
            label = self.flip_map.get(label, label)

        return stacked, label


# %%
# ──────────────────────────────────────────────────────────────────────────────
# BUILD DATALOADERS
# ──────────────────────────────────────────────────────────────────────────────
dataset    = SubwayDataset(dataset_dir=DATASET_DIR, transform=frame_transform)
val_size   = int(VAL_SPLIT * len(dataset))
train_size = len(dataset) - val_size
train_set, val_set = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"Dataset: {len(dataset)} samples  "
      f"(train={train_size}, val={val_size})  "
      f"Stack size: {STACK_SIZE}")

# %%
# ──────────────────────────────────────────────────────────────────────────────
# MODEL, LOSS, OPTIMISER
# ──────────────────────────────────────────────────────────────────────────────
model     = SubwayCNN(in_channels=STACK_SIZE, num_actions=5).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# %%
# ──────────────────────────────────────────────────────────────────────────────
# TRAINING LOOP
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        # ── train ────────────────────────────────────────────────────────────
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1:02d}/{EPOCHS}", leave=False)
        for images, labels in loop:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct    += (outputs.argmax(1) == labels).sum().item()
            total      += labels.size(0)

            loop.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total:.3f}")

        train_acc = correct / total

        # ── validate ─────────────────────────────────────────────────────────
        model.eval()
        val_correct, val_total = 0, 0

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="  validating", leave=False):
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total   += labels.size(0)

        val_acc = val_correct / val_total

        print(
            f"Epoch {epoch+1:02d}/{EPOCHS} | "
            f"Loss: {total_loss/len(train_loader):.4f} | "
            f"Train acc: {train_acc:.3f} | "
            f"Val acc: {val_acc:.3f}"
        )

        # ── checkpoint ───────────────────────────────────────────────────────
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch":            epoch + 1,
                "model_state":      model.state_dict(),
                "optimizer_state":  optimizer.state_dict(),
                "val_acc":          val_acc,
                "in_channels":      STACK_SIZE,
                "num_actions":      5,
                "stack_size":       STACK_SIZE,
            }, SAVE_PATH)
            print(f"  [SAVED] best val acc: {val_acc:.3f}")

    print(f"\nDone. Best val acc: {best_val_acc:.3f} → {SAVE_PATH}")
