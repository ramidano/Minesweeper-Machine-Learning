import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MinesweeperDataset(Dataset):
    def __init__(self, filepath):
        data = np.load(filepath)
        self.boards = torch.LongTensor(data['boards'] + 1)
        self.mines = torch.FloatTensor(data['mines'])
        print(f"Loaded {len(self.boards)} samples from {filepath}")
    
    def __len__(self):
        return len(self.boards)

    def __getitem__(self, idx):
        return self.boards[idx], self.mines[idx]

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(10, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1)
        )

    def forward(self, x):
        x = nn.functional.one_hot(x, num_classes=10).permute(0, 3, 1, 2).float()
        return self.cnn(x)


def train():
    print("Using device:", device)
    train_loader = DataLoader(MinesweeperDataset("data/train.npz"), batch_size=128, shuffle=True, num_workers=2)
    val_loader = DataLoader(MinesweeperDataset("data/val.npz"), batch_size=128, shuffle=False, num_workers=2)
    model = CNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()
    for epoch in range(5):
        model.train()
        total_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs.to(device))
            loss = criterion(outputs, targets.to(device).unsqueeze(1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        model.eval()
        val_loss = 0
        val_acc = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs.to(device))
                loss = criterion(outputs, targets.to(device).unsqueeze(1))
                val_loss += loss.item()
                val_acc += ((outputs > 0).float() == targets.to(device).unsqueeze(1)).float().sum() / torch.numel(targets.to(device).unsqueeze(1))
        print(f"Epoch {epoch+1}: Train Loss={total_loss / len(train_loader):.4f} | Val Loss={val_loss / len(val_loader):.4f} | Val Acc={val_acc / len(val_loader):.4f}")
    torch.save(model.state_dict(), "model/minesweeper_small_model.pth")
    print("Model saved to minesweeper_small_model.pth")

if __name__ == "__main__":
    train()