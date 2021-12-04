import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from pathlib import Path

plots_folder = Path.cwd() / "plots"
plots_folder.mkdir(exist_ok=True)

field_lim=0.5

colors = (
    [(0.4*q+0.6, q*0.6, 0) for q in np.linspace(1, 0, 64)] +
    [(0.6*q, 0, 0) for q in np.linspace(1, 0, 64)] +
    [ (0, q, q) for q in np.linspace(0, 1, 128) ]
)
cmap = ListedColormap(colors)

data = pd.read_csv("test")

def flatten(row):
    pos_cols = row[list(filter(lambda cname: cname[0] == "q", row.index))]
    pos = [ cname[2:].split('_') for cname in pos_cols.index]
    pos = [ (int(i), int(j)) for (i, j) in pos]
    height = max(map(lambda x: x[0], pos))
    width = max(map(lambda x: x[1], pos))
    vals = np.zeros(shape=(height+1, width+1))
    for i, j in pos:
        vals[i, j] = float(pos_cols[f"q_{i}_{j}"])
    return vals

print(len(data))
for i in range(len(data)):
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(flatten(data.iloc[i]), vmin=-field_lim, vmax=field_lim, cmap=cmap)
    fig.savefig(plots_folder / f"{i}.png")
    plt.close(fig)

# ax.set_xlim(-1.5, 1.5)
# ax.set_ylim(-0.0001, 0.0005)
# ax.plot(data["q1"], data["q2"])
# ax.plot(data["q4"], data["q5"])
