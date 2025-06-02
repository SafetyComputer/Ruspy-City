import numpy as np
from utils import *


loaded_data = np.load("./data/processed_data.npz")
planes, best_moves, evals = loaded_data['planes'], loaded_data['best_moves'], loaded_data['evals']


# print(planes[:8])
# print(best_moves[:8])
# print(evals[:8])


for i in range(8, 16):
    print(f"Best move {convert_index_to_move(best_moves[i - 8])}")
    print(f"Plane {i}:")
    print(planes[i,[2, 3]])
    print(f"Best move: {convert_index_to_move(best_moves[i])}")
    print(f"Evaluation: {evals[i]}")
    print()