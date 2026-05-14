import pickle

import matplotlib.pyplot as plt
import numpy as np

plt.style.use("~/plotStyle.mplstyle")
HType = "ToricOpenPQUV3"
d = 2
ansatz = "Conv2D_nl=11_oct"
L = 8


filename = f"logs/{HType.lower()}/d={d}/{ansatz}_L={L}.log"

with open(filename, "rb") as f:
    nlls = pickle.load(f)
nlls = np.array(nlls)
print(nlls[-1, 0])
plt.plot(nlls[:, -1], label=ansatz)
plt.title(rf"L = {L} $p_{{err}}= 0.19$ layers = 11")
plt.ylabel("Cross Entropy")
plt.xlabel("Training Iterations")
plt.savefig("images/trainingPlot.png", dpi=500, bbox_inches="tight")
