import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

rows = 2
columns = 4

fig = plt.figure(figsize=(10, 8))

img = cv.imread("poke.png")

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

u, s, v = np.linalg.svd(gray, full_matrices=False)

rank_values = [50, 100, 150, 200, 250, 300, 350, gray.shape[0]]

for i, rank in enumerate(rank_values):
    low_rank = u[:, :rank] @ np.diag(s[:rank]) @ v[:rank, :]
    ax = fig.add_subplot(rows, columns, i+1)
    ax.set_title(f"Rank {rank} Approximation")
    plt.imshow(low_rank)

plt.show()