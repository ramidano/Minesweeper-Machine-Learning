import random
from collections import deque
from dataclasses import dataclass
import time

@dataclass
class Cell:
    is_mine: bool = False
    is_open: bool = False
    is_flagged: bool = False
    adjacent: int = 0
    exploded: bool = False

class Board:
    def __init__(self, side: int, mine_count: int, seed: int | None = None):
        self.side = side
        self.mine_count = mine_count
        self.grid = [[Cell() for _ in range(side)] for _ in range(side)]
        self._placed = False
        if seed is not None:
            random.seed(seed)

    def in_bounds(self, x, y):
        return 0 <= x < self.side and 0 <= y < self.side

    def neighbors(self, x, y):
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if self.in_bounds(nx, ny):
                    yield nx, ny

    def place_mines(self, first_click: tuple[int, int] | None = None):
        """Place mines randomly. Optionally avoid first_click cell and its neighbors."""
        cells = [(x, y) for x in range(self.side) for y in range(self.side)]
        if first_click is not None:
            fx, fy = first_click
            forbidden = {(fx, fy)} | { (nx, ny) for nx,ny in self.neighbors(fx, fy) }
            cells = [c for c in cells if c not in forbidden]
        chosen = random.sample(cells, self.mine_count)
        for x, y in chosen:
            self.grid[x][y].is_mine = True
        # compute adjacent counts
        for x in range(self.side):
            for y in range(self.side):
                if not self.grid[x][y].is_mine:
                    self.grid[x][y].adjacent = sum(1 for nx, ny in self.neighbors(x, y) if self.grid[nx][ny].is_mine)
        self._placed = True

    def reveal(self, x, y):
        """Reveal cell at (x,y). Returns ('ok', cells_revealed_count) or ('mine', exploded_cell) or ('already',0)."""
        cell = self.grid[x][y]
        if cell.is_open or cell.is_flagged:
            return ('already', 0)
        if not self._placed:
            self.place_mines(first_click=(x, y))
        if cell.is_mine:
            cell.exploded = True
            cell.is_open = True
            return ('mine', (x, y))
        # flood fill if zero, otherwise open one
        revealed = 0
        queue = deque()
        queue.append((x, y))
        while queue:
            cx, cy = queue.popleft()
            c = self.grid[cx][cy]
            if c.is_open or c.is_flagged:
                continue
            c.is_open = True
            revealed += 1
            if c.adjacent == 0:
                for nx, ny in self.neighbors(cx, cy):
                    neigh = self.grid[nx][ny]
                    if not neigh.is_open and not neigh.is_flagged and not neigh.is_mine:
                        queue.append((nx, ny))
        return ('ok', revealed)

    def toggle_flag(self, x, y):
        c = self.grid[x][y]
        if c.is_open:
            return False
        c.is_flagged = not c.is_flagged
        return True

    def chord(self, x, y):
        """
        If the cell at (x,y) is open and its adjacent flag count equals its adjacent number,
        reveal all non-flagged neighbors. Returns list of revealed coordinates or
        ('mine', exploded_cell) if a mine was revealed by a mistaken chord.
        """
        cell = self.grid[x][y]
        if not cell.is_open or cell.adjacent == 0:
            return []
        flag_count = sum(1 for nx, ny in self.neighbors(x, y) if self.grid[nx][ny].is_flagged)
        if flag_count != cell.adjacent:
            return []
        revealed_coords = []
        for nx, ny in self.neighbors(x, y):
            neighbor = self.grid[nx][ny]
            if not neighbor.is_flagged and not neighbor.is_open:
                if neighbor.is_mine:
                    neighbor.exploded = True
                    neighbor.is_open = True
                    return ('mine', (nx, ny))
                res, _ = self.reveal(nx, ny)  # reveal will flood-fill as needed
                # reveal might have already added many cells; we won't track duplicates here
                revealed_coords.append((nx, ny))
        return revealed_coords

    def reveal_all_mines(self):
        """Mark all mines as open (for endgame display)."""
        for x in range(self.side):
            for y in range(self.side):
                c = self.grid[x][y]
                if c.is_mine:
                    c.is_open = True

    def flags_left(self):
        flags = sum(1 for row in self.grid for c in row if c.is_flagged)
        return max(0, self.mine_count - flags)

    def opened_count(self):
        return sum(1 for row in self.grid for c in row if c.is_open)

    def check_win(self):
        total_cells = self.side * self.side
        opened = self.opened_count()
        return opened >= total_cells - self.mine_count

class Game:
    DIFFICULTIES = {
        1: (10, 10),
        2: (18, 40),
        3: (24, 99),
    }

    def __init__(self, difficulty=1, seed=None):
        if difficulty not in self.DIFFICULTIES:
            difficulty = 1
        side, mines = self.DIFFICULTIES[difficulty]
        self.board = Board(side, mines, seed=seed)
        self.difficulty = difficulty
        self.started = False
        self.lost = False
        self.won = False
        self.start_time = None
        self.end_time = None

    def start_timer(self):
        import time
        if not self.started:
            self.start_time = time.time()
            self.started = True

    def reveal(self, x, y):
        if self.lost or self.won:
            return
        if not self.started:
            self.start_timer()
        res, data = self.board.reveal(x, y)
        if res == 'mine':
            self.lost = True
            self.end_time = time.time()
            self.board.reveal_all_mines()
            return ('lost', data)
        if self.board.check_win():
            self.won = True
            self.end_time = time.time()
            self.board.reveal_all_mines()
            return ('won', None)
        return ('ok', data)

    def toggle_flag(self, x, y):
        if self.lost or self.won:
            return False
        return self.board.toggle_flag(x, y)

    def chord(self, x, y):
        if self.lost or self.won:
            return
        if not self.started:
            return
        res = self.board.chord(x, y)
        if isinstance(res, tuple) and res[0] == 'mine':
            self.lost = True
            self.end_time = time.time()
            self.board.reveal_all_mines()
            return ('lost', res[1])
        if self.board.check_win():
            self.won = True
            self.end_time = time.time()
            self.board.reveal_all_mines()
            return ('won', None)
        return ('ok', res)

    def flags_left(self):
        return self.board.flags_left()

    def elapsed_seconds(self):
        import time
        if not self.started:
            return 0
        end = self.end_time if self.end_time else time.time()
        return end - self.start_time

    def reset(self, difficulty=None, seed=None):
        if difficulty is not None:
            self.difficulty = difficulty
        side, mines = self.DIFFICULTIES[self.difficulty]
        self.board = Board(side, mines, seed=seed)
        self.started = False
        self.lost = False
        self.won = False
        self.start_time = None
        self.end_time = None