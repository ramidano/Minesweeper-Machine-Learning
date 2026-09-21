import sys
import os
import time
import numpy as np
import torch
import torch.nn as nn
import pygame
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from game.minesweeper_engine import Game
import game.minesweeper_pygame as ms_gui

ai_delay = 0.0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        return self.cnn(x)

class MinesweeperAI:
    def __init__(self, model_path):
        self.device = device
        self.model = CNN().to(self.device)
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            print("Loaded model:", model_path)
        else:
            print(model_path, "not found")
            self.model = None

    def get_board_tensor(self, game):
        side = game.board.side
        board_np = np.full((side, side), -1, dtype=np.int8)
        for r in range(side):
            for c in range(side):
                cell = game.board.grid[r][c]
                if cell.is_open:
                    board_np[r, c] = cell.adjacent
                elif cell.is_flagged:
                    board_np[r, c] = -1 
                else:
                    board_np[r, c] = -1
        board_indices = board_np + 1 
        one_hot = np.eye(10)[board_indices] 
        features = np.transpose(one_hot, (2, 0, 1)).astype(np.float32)
        tensor = torch.from_numpy(features).unsqueeze(0).to(self.device)
        return tensor

    def flip_ensemble(self, input_tensor):
        if self.model is None: return np.zeros((18,18))
        with torch.no_grad():
            logits_sum = self.model(input_tensor)
            inputs_flip = torch.flip(input_tensor, [3])
            logits_sum += torch.flip(self.model(inputs_flip), [3])
            for k in [1, 2, 3]:
                logits_sum += torch.rot90(self.model(torch.rot90(input_tensor, k, [2, 3])), -k, [2, 3])
                logits_sum += torch.flip(torch.rot90(self.model(torch.rot90(inputs_flip, k, [2, 3])), -k, [2, 3]), [3])
            return torch.sigmoid(logits_sum / 8.0).squeeze().cpu().numpy()

    def get_best_move(self, game):
        probs = self.flip_ensemble(self.get_board_tensor(game))
        side = game.board.side
        valid_mask = np.zeros((side, side), dtype=bool)
        for r in range(side):
            for c in range(side):
                cell = game.board.grid[r][c]
                if not cell.is_open and not cell.is_flagged:
                    valid_mask[r, c] = True
        if not np.any(valid_mask):
            return None, 0.0, "none"
        # 1. Reveal safe tiles (< 3%)
        safe_indices = np.where((probs < 0.03) & valid_mask)
        if len(safe_indices[0]) > 0:
            safe_probs = np.where(valid_mask, probs, np.inf)
            arg_min = np.unravel_index(np.argmin(safe_probs), safe_probs.shape)
            return arg_min, probs[arg_min], "neural_safe"
        # 2. Flag definite mines (> 95%)
        flag_indices = np.where((probs > 0.95) & valid_mask)
        if len(flag_indices[0]) > 0:
            r, c = flag_indices[0][0], flag_indices[1][0]
            if not game.board.grid[r][c].is_flagged:
                return (r, c), probs[r, c], "neural_flag"
        # 3. Guessing (Logic failed, no safe moves)
        safe_probs = np.where(valid_mask, probs, np.inf)
        min_prob = np.min(safe_probs)
        arg_min = np.unravel_index(np.argmin(safe_probs), safe_probs.shape)
        return arg_min, min_prob, "neural_guess"

class AIEnhancedGUI(ms_gui.MinesweeperGUI):
    def __init__(self, difficulty=2):
        super().__init__(difficulty)
        self.ai = MinesweeperAI("minesweeper_small_model.pth")
        self.ai_active = False
        self.last_ai_move = 0
        self.current_ai_prob = 0.0
        self.current_ai_action = "Ready"

    def ai_step(self):
        if self.game.lost or self.game.won:
            return
        move, prob, action = self.ai.get_best_move(self.game)
        self.current_ai_prob = prob
        self.current_ai_action = action
        if move:
            if "reveal" in action or "guess" in action or "safe" in action:
                self.game.reveal(move[0], move[1])
            elif "flag" in action:
                self.game.toggle_flag(move[0], move[1])

    def _handle_events(self):
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                self.running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    self.running = False
                elif ev.key == pygame.K_r:
                    self.game.reset(self.game.difficulty)
                    self._recalc_window()
                    self.screen = pygame.display.set_mode((self.w, self.h))
                    self.ai_active = False
                    self.current_ai_action = "Ready"
                elif ev.key == pygame.K_a:
                    self.ai_active = not self.ai_active
                elif ev.key == pygame.K_RIGHT:
                    self.ai_step()
            elif ev.type == pygame.MOUSEBUTTONDOWN:
                mx, my = ev.pos
                gx = (mx - ms_gui.MARGIN) // self.cell_size
                gy = (my - ms_gui.TOP_BAR) // self.cell_size
                if 0 <= gx < self.game.board.side and 0 <= gy < self.game.board.side:
                    if ev.button == 1:
                        self.game.reveal(gx, gy)
                    elif ev.button == 3:
                        self.game.toggle_flag(gx, gy)

    def run(self):
        while self.running:
            self.clock.tick(ms_gui.FPS)
            self._handle_events()
            now = time.time()
            if self.ai_active:
                if now - self.last_ai_move > ai_delay:
                    self.ai_step()
                    self.last_ai_move = now
            self._draw()
            color = (0, 150, 0) if self.ai_active else (50, 50, 50)
            txt_str = f"AI: {self.current_ai_action}"
            if self.current_ai_prob > 0:
                txt_str += f" ({self.current_ai_prob:.2f})"
            txt_surf = self.font_small.render(txt_str, True, color)
            self.screen.blit(txt_surf, (self.w//2-50, ms_gui.TOP_BAR - 15))
            pygame.display.flip()
        pygame.quit()

if __name__ == "__main__":
    gui = AIEnhancedGUI(difficulty=2)
    gui.run()