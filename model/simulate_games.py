import sys
import os
import time
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
sys.path.append('../ms')
from game.minesweeper_engine import Game

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

class BatchSimulator:
    def __init__(self, model_path, batch_size):
        self.batch_size = batch_size
        self.device = device
        self.model = ResNet().to(self.device)
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            print("Loaded model:", model_path)
        else:
            raise FileNotFoundError(model_path, "not found")
        print(f"Initializing {batch_size} concurrent games...")
        self.games = [Game(difficulty=2) for _ in range(batch_size)]
        for g in self.games:
            center = g.board.side // 2
            g.reveal(center, center)
        self.stats = {
            "wins": 0,
            "losses": 0,
            "games_played": 0
        }
        self.failures = {
            "boards": [],
            "moves": [],
            "actions": [],
            "probs": []
        }

    def get_batch_tensors(self):
        side = self.games[0].board.side
        batch_np = np.full((self.batch_size, side, side), -1, dtype=np.int8)
        for i, game in enumerate(self.games):
            grid = game.board.grid
            for r in range(side):
                for c in range(side):
                    cell = grid[r][c]
                    if cell.is_open:
                        batch_np[i, r, c] = cell.adjacent
        batch_indices = batch_np + 1
        one_hot = np.eye(10)[batch_indices]
        features = np.transpose(one_hot, (0, 3, 1, 2)).astype(np.float32)
        return torch.from_numpy(features).to(self.device)

    def predict_batch_tta(self, inputs):
        with torch.no_grad():
            logits_sum = self.model(inputs)
            logits_sum += torch.flip(self.model(torch.flip(inputs, [3])), [3])
            for k in [1, 2, 3]:
                logits_sum += torch.rot90(self.model(torch.rot90(inputs, k, [2, 3])), -k, [2, 3])
                logits_sum += torch.flip(torch.rot90(self.model(torch.rot90(torch.flip(inputs, [3]), k, [2, 3])), -k, [2, 3]), [3])
            return torch.sigmoid(logits_sum / 8.0).squeeze(1).cpu().numpy()

    def run(self, total_games):
        pbar = tqdm(total=total_games, unit="games")
        while self.stats["games_played"] < total_games:
            tensor_batch = self.get_batch_tensors()
            probs_batch = self.predict_batch_tta(tensor_batch)
            for i in range(self.batch_size):
                game = self.games[i]
                probs = probs_batch[i]
                side = game.board.side
                grid = game.board.grid
                valid_mask = np.zeros((side, side), dtype=bool)
                for r in range(side):
                    for c in range(side):
                        if not grid[r][c].is_open and not grid[r][c].is_flagged:
                            valid_mask[r, c] = True
                if not np.any(valid_mask):
                    game.reset(difficulty=2)
                    center = side // 2
                    game.reveal(center, center)
                    continue
                move = None
                action = ""
                prob = 0.0
                # 1. Reveal safe tiles (< 3%)
                if move is None:
                    safe_indices = np.where((probs < 0.03) & valid_mask)
                    if len(safe_indices[0]) > 0:
                        safe_probs = np.where(valid_mask, probs, np.inf)
                        idx = np.argmin(safe_probs)
                        r, c = np.unravel_index(idx, (side, side))
                        move = (r, c)
                        action = "neural_safe"
                        prob = probs[r, c]
                # 2. Flag definite mines (> 95%)
                flag_indices = np.where((probs > 0.95) & valid_mask)
                if len(flag_indices[0]) > 0:
                    r, c = flag_indices[0][0], flag_indices[1][0]
                    if not grid[r][c].is_flagged:
                        move = (r, c)
                        action = "neural_flag"
                        prob = probs[r, c]
                # 3. Guessing (Logic failed, no safe moves)
                if move is None:
                    safe_probs = np.where(valid_mask, probs, np.inf)
                    idx = np.argmin(safe_probs)
                    r, c = np.unravel_index(idx, (side, side))
                    move = (r, c)
                    action = "neural_guess"
                    prob = probs[r, c]
                # SAVE STATE BEFORE MOVE (For Failure Record)
                if action != "neural_flag":
                    board_snapshot = np.full((side, side), -1, dtype=np.int8)
                    for r_ in range(side):
                        for c_ in range(side):
                            if grid[r_][c_].is_open:
                                board_snapshot[r_, c_] = grid[r_][c_].adjacent
                res = None
                if "flag" in action:
                    game.toggle_flag(move[0], move[1])
                else:
                    res = game.reveal(move[0], move[1])
                game_over = False
                if game.won:
                    self.stats["wins"] += 1
                    game_over = True
                elif game.lost or (res and isinstance(res, tuple) and res[0] == 'lost'):
                    self.stats["losses"] += 1
                    game_over = True
                    # RECORD FAILURE
                    self.failures["boards"].append(board_snapshot)
                    self.failures["moves"].append(np.array(move, dtype=np.int8))
                    self.failures["actions"].append(action)
                    self.failures["probs"].append(float(prob))
                if game_over:
                    self.stats["games_played"] += 1
                    pbar.update(1)
                    # Periodic Save
                    if self.stats["games_played"] % 10_000 == 0:
                        self.save_failures()
                    # Reset this game slot for the next round
                    game.reset(difficulty=2)
                    center = side // 2
                    game.reveal(center, center)
        pbar.close()
        self.save_failures()
        print("\nSimulation Complete.")
        print(f"Wins: {self.stats['wins']}")
        print(f"Losses: {self.stats['losses']}")
        print(f"Win Rate: {self.stats['wins'] / self.stats['games_played'] * 100:.2f}%")

    def save_failures(self):
        np.savez_compressed(
            "model/failures.npz",
            boards=np.array(self.failures["boards"], dtype=np.int8),
            moves=np.array(self.failures["moves"], dtype=np.int8),
            actions=np.array(self.failures["actions"]),
            probs=np.array(self.failures["probs"], dtype=np.float32)
        )

if __name__ == "__main__":
    sim = BatchSimulator("minesweeper_model.pth", batch_size=2048)
    try:
        sim.run(10000)
    except KeyboardInterrupt:
        print("\nSimulation stopped by user. Saving data...")
        sim.save_failures()