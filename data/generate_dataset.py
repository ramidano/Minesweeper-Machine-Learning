import numpy as np
import random
import multiprocessing as mp
from tqdm import tqdm
import os
import sys
sys.path.append('../ms')
from game.minesweeper_engine import Game

# generate one sample
def generate_sample(_=None):
    game = Game(difficulty=2)
    side = game.board.side
    # click somewhere random
    game.reveal(random.randrange(side), random.randrange(side))
    # reveal some random safe tiles
    safe_tiles = [(r, c) for r in range(side) for c in range(side) if not game.board.grid[r][c].is_mine and not game.board.grid[r][c].is_open]
    random.shuffle(safe_tiles)
    for i in range(min(8, len(safe_tiles))):
        x, y = safe_tiles[i]
        game.reveal(x, y)
    # make board and mines arrays
    board = np.zeros((side, side), dtype=np.int8)
    mines = np.zeros((side, side), dtype=np.int8)
    for r in range(side):
        for c in range(side):
            cell = game.board.grid[r][c]
            board[r, c] = cell.adjacent if cell.is_open else -1
            mines[r, c] = 1 if cell.is_mine else 0
    return board, mines

# generates dataset for train/val/test
def generate_split(total_samples, save_path):
    print("Generating", save_path)
    job_iter = [None] * total_samples
    boards_list = []
    mines_list = []
    written_chunks = []
    with mp.Pool(12) as pool:
        for result in tqdm(pool.imap_unordered(generate_sample, job_iter, chunksize=1000), total=total_samples):
            b, m = result
            boards_list.append(b)
            mines_list.append(m)
            # write to temp if chunk is full
            if len(boards_list) == 10000:
                tmp_name = save_path + ".part" + str(len(written_chunks)) + ".npz"
                np.savez_compressed(tmp_name, boards=np.array(boards_list, dtype=np.int8), mines=np.array(mines_list, dtype=np.int8))
                written_chunks.append(tmp_name)
                boards_list = []
                mines_list = []
    # write last chunk
    if boards_list:
        tmp_name = save_path + ".part" + str(len(written_chunks)) + ".npz"
        np.savez_compressed(tmp_name, boards=np.array(boards_list, dtype=np.int8), mines=np.array(mines_list, dtype=np.int8))
        written_chunks.append(tmp_name)
    # merge into big npz
    print("Merging into", save_path)
    boards_all = []
    mines_all = []
    for file in written_chunks:
        with np.load(file) as data:
            boards_all.append(data["boards"])
            mines_all.append(data["mines"])
    boards_all = np.vstack(boards_all)
    mines_all = np.vstack(mines_all)
    np.savez_compressed(save_path, boards=boards_all, mines=mines_all)
    print("Saved", save_path, "with shape:", boards_all.shape)
    # clean
    for file in written_chunks:
        try:
            os.remove(file)
        except PermissionError:
            print("can't delete temp file:", file)

if __name__ == "__main__":
    generate_split(1000000, "data/train.npz")
    generate_split(100000, "data/val.npz")
    generate_split(100000, "data/test.npz")
    print("done")