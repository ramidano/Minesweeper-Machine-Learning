import sys
import pygame
import traceback
import argparse
import sys
sys.path.append('../ms')
from game.minesweeper_engine import Game

FPS = 60
CELL_SIZE = 28
MARGIN = 20
TOP_BAR = 60
FONT_NAME = None
# colors
BG = (192, 192, 192)
GRID_BG = (192, 192, 192)
CELL_COVER = (100, 100, 100)
CELL_OPEN = (225, 225, 225)
FLAG_COLOR = (252, 55, 5)
MINE_COLOR = (252, 55, 5)
TEXT_COLOR = (10, 10, 10)
HIGHLIGHT = (200, 200, 255)
NUM_COLORS = {
    1: (25, 25, 200),
    2: (25, 120, 25),
    3: (200, 25, 25),
    4: (25, 25, 120),
    5: (120, 25, 25),
    6: (25, 120, 120),
    7: (30, 30, 30),
    8: (100, 100, 100),
}

# ---------- Helpers ----------
def draw_text_center(surface, text, rect, font, color=(0,0,0)):
    txt = font.render(text, True, color)
    tr = txt.get_rect(center=rect.center)
    surface.blit(txt, tr)
def draw_text_right(surface, text, rect, font, color=(0,0,0)):
    txt = font.render(text, True, color)
    tr = txt.get_rect(midright=rect.midright) 
    surface.blit(txt, tr)
# ---------- Main GUI ----------
class MinesweeperGUI:
    def __init__(self, difficulty=1):
        pygame.init()
        pygame.display.set_caption("Minesweeper (Pygame)")
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont(FONT_NAME, 16)
        self.font_mid = pygame.font.SysFont(FONT_NAME, 28)
        self.font_big = pygame.font.SysFont(FONT_NAME, 48, bold=True)

        self.game = Game(difficulty=difficulty)
        self.cell_size = CELL_SIZE
        self._recalc_window()
        self.screen = pygame.display.set_mode((self.w, self.h))
        self.running = True

    def _recalc_window(self):
        side = self.game.board.side
        if side >= 24:
            self.cell_size = 22
        elif side >= 18:
            self.cell_size = 26
        else:
            self.cell_size = CELL_SIZE
        self.w = MARGIN * 2 + self.cell_size * side
        self.h = TOP_BAR + MARGIN + self.cell_size * side

    def run(self):
        while self.running:
            self.clock.tick(FPS)
            self._handle_events()
            self._draw()
            pygame.display.flip()
        pygame.quit()

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
                elif ev.key == pygame.K_1:
                    self.game.reset(difficulty=1)
                    self._recalc_window()
                    self.screen = pygame.display.set_mode((self.w, self.h))
                elif ev.key == pygame.K_2:
                    self.game.reset(difficulty=2)
                    self._recalc_window()
                    self.screen = pygame.display.set_mode((self.w, self.h))
                elif ev.key == pygame.K_3:
                    self.game.reset(difficulty=3)
                    self._recalc_window()
                    self.screen = pygame.display.set_mode((self.w, self.h))
            elif ev.type == pygame.MOUSEBUTTONDOWN:
                mx, my = ev.pos
                side = self.game.board.side
                gx = (mx - MARGIN) // self.cell_size
                gy = (my - TOP_BAR) // self.cell_size
                if 0 <= gx < side and 0 <= gy < side:
                    if ev.button == 1:
                        cell = self.game.board.grid[gx][gy]
                        if not cell.is_open:
                            self.game.reveal(gx, gy)
                        else:
                            self.game.chord(gx, gy)
                    elif ev.button == 3:
                        self.game.toggle_flag(gx, gy)

    def _draw(self):
        self.screen.fill(BG)
        pygame.draw.rect(self.screen, GRID_BG, (0,0, self.w, TOP_BAR))
        mines_left = self.game.flags_left()
        timer = self.game.elapsed_seconds()
        left_text = f"Mines: {mines_left}"
        mid_text = f"Time: {timer:03.1f}"
        right_text = "[R] New game"
        rect_left = pygame.Rect(0, 10, 150, TOP_BAR-20)
        rect_mid = pygame.Rect(self.w//2 - 75, 10, 150, TOP_BAR-20)
        rect_right = pygame.Rect(self.w - 180, 10, 150, TOP_BAR-20)
        draw_text_center(self.screen, left_text, rect_left, self.font_mid, TEXT_COLOR)
        draw_text_center(self.screen, mid_text, rect_mid, self.font_mid, TEXT_COLOR)
        draw_text_right(self.screen, right_text, rect_right, self.font_mid, TEXT_COLOR)
        side = self.game.board.side
        grid_rect = pygame.Rect(MARGIN, TOP_BAR, self.cell_size*side, self.cell_size*side)
        pygame.draw.rect(self.screen, GRID_BG, grid_rect)
        for x in range(side):
            for y in range(side):
                cx = MARGIN + x * self.cell_size
                cy = TOP_BAR + y * self.cell_size
                cell = self.game.board.grid[x][y]
                cell_rect = pygame.Rect(cx, cy, self.cell_size, self.cell_size)

                if cell.is_open:
                    pygame.draw.rect(self.screen, CELL_OPEN, cell_rect)
                    pygame.draw.rect(self.screen, TEXT_COLOR, cell_rect, 1)
                    if cell.is_mine:
                        if cell.exploded:
                            pygame.draw.circle(self.screen, (255, 0, 0), cell_rect.center, self.cell_size//3)
                        else:
                            pygame.draw.circle(self.screen, MINE_COLOR, cell_rect.center, self.cell_size//4)
                    elif cell.adjacent > 0:
                        color = NUM_COLORS.get(cell.adjacent, TEXT_COLOR)
                        txt = self.font_mid.render(str(cell.adjacent), True, color)
                        tr = txt.get_rect(center=cell_rect.center)
                        self.screen.blit(txt, tr)
                else:
                    pygame.draw.rect(self.screen, CELL_COVER, cell_rect)
                    pygame.draw.rect(self.screen, TEXT_COLOR, cell_rect, 1)
                    if cell.is_flagged:
                        tri = [
                            (cx + self.cell_size*0.2, cy + self.cell_size*0.8),
                            (cx + self.cell_size*0.5, cy + self.cell_size*0.3),
                            (cx + self.cell_size*0.8, cy + self.cell_size*0.8),
                        ]
                        pygame.draw.polygon(self.screen, FLAG_COLOR, tri)

        # grid lines
        for i in range(side+1):
            # vertical
            x = MARGIN + i * self.cell_size
            pygame.draw.line(self.screen, GRID_BG, (x, TOP_BAR), (x, TOP_BAR + side*self.cell_size), 1)
            # horizontal
            y = TOP_BAR + i * self.cell_size
            pygame.draw.line(self.screen, GRID_BG, (MARGIN, y), (MARGIN + side*self.cell_size, y), 1)

        # Draw CENTER SCREEN Game Over / Victory Message
        if self.game.lost or self.game.won:
            msg = "VICTORY!" if self.game.won else "GAME OVER"
            color = (0, 180, 0) if self.game.won else (220, 0, 0)
            text_surf = self.font_big.render(msg, True, color)
            text_rect = text_surf.get_rect(center=(self.w//2, self.h//2 + TOP_BAR//2))
            bg_rect = text_rect.inflate(40, 30)
            s = pygame.Surface((bg_rect.width, bg_rect.height))
            s.set_alpha(200)
            s.fill((255, 255, 255))
            self.screen.blit(s, bg_rect.topleft)
            pygame.draw.rect(self.screen, color, bg_rect, 4)
            self.screen.blit(text_surf, text_rect)


# --------- Run entrypoint ----------
def parse_args():
    p = argparse.ArgumentParser(description="Minesweeper Pygame GUI")
    p.add_argument("--difficulty", "-d", type=int, choices=(1,2,3), default=1,
                   help="Difficulty: 1=10x10(10), 2=18x18(40), 3=24x24(99)")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    try:
        gui = MinesweeperGUI(difficulty=args.difficulty)
        gui.run()
    except Exception:
        traceback.print_exc()
        pygame.quit()
        sys.exit(1)