import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MinesweeperDataset(Dataset):
    def __init__(self, filepath, augment=False):
        data = np.load(filepath)
        self.boards = data['boards']
        self.mines = data['mines']
        self.augment = augment
        print("Loaded", len(self.boards), "samples from", filepath)

    def __len__(self):
        return len(self.boards)

    def __getitem__(self, idx):
        board = self.boards[idx]
        mines = self.mines[idx]
        if self.augment:
            k = np.random.randint(0, 4)
            board = np.rot90(board, k)
            mines = np.rot90(mines, k)
            if np.random.random() > 0.5:
                board = np.fliplr(board)
                mines = np.fliplr(mines)
        board_indices = board + 1
        one_hot = np.eye(10)[board_indices] 
        features = np.transpose(one_hot, (2, 0, 1)).astype(np.float32)
        target = np.expand_dims(mines, axis=0).astype(np.float32)
        return torch.from_numpy(features.copy()), torch.from_numpy(target.copy())

class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.res = nn.Sequential(nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(channels),
                                 nn.ReLU(),
                                 nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(channels))
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.res(x)
        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.start = nn.Sequential(nn.Conv2d(10, 192, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(192),
                                   nn.ReLU())
        self.blocks = nn.Sequential(ResBlock(192),
                                    ResBlock(192),
                                    ResBlock(192),
                                    ResBlock(192),
                                    ResBlock(192),
                                    ResBlock(192),
                                    ResBlock(192),
                                    ResBlock(192))
        self.end = nn.Conv2d(192, 1, kernel_size=1)

    def forward(self, x):
        x = self.start(x)
        x = self.blocks(x)
        x = self.end(x)
        return x


def flip_ensemble(model, inputs):
    logits_sum = model(inputs)
    inputs_flip = torch.flip(inputs, [3])
    logits_sum += torch.flip(model(inputs_flip), [3])
    for k in [1, 2, 3]:
        logits_sum += torch.rot90(model(torch.rot90(inputs, k, [2, 3])), -k, [2, 3])
        logits_sum += torch.flip(torch.rot90(model(torch.rot90(inputs_flip, k, [2, 3])), -k, [2, 3]), [3])
    return logits_sum / 8.0

def train():
    print("Using device:", device)    
    train_loader = DataLoader(MinesweeperDataset("data/train.npz", augment=True), batch_size=1024, shuffle=True, num_workers=24, pin_memory=True)
    val_loader = DataLoader(MinesweeperDataset("data/val.npz", augment=False), batch_size=1024, shuffle=False, num_workers=24, pin_memory=True)
    model = ResNet().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda')
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 20)
    for epoch in range(20):
        model.train()
        total_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                outputs = model(inputs.to(device))
                loss = criterion(outputs, targets.to(device))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            scheduler.step()
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                with torch.amp.autocast('cuda'):
                    outputs = flip_ensemble(model, inputs.to(device))
                    loss = criterion(outputs, targets.to(device))
                val_loss += loss.item()
                val_acc += ((outputs > 0).float() == targets.to(device)).float().sum() / torch.numel(targets.to(device))
        print(f"Epoch {epoch+1}: Train Loss={total_loss / len(train_loader):.4f} | Val Loss={val_loss / len(val_loader):.4f} | Val Acc={val_acc / len(val_loader):.4f}")
    torch.save(model.state_dict(), "model/minesweeper_model.pth")
    print("Model saved to minesweeper_model.pth")

if __name__ == "__main__":
    train()