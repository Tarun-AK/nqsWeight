import numpy as np
import pandas as pd
import ray
from ray.experimental import tqdm_ray
from tensorNetwork import TN

ray.init(ignore_reinit_error=True)

RemoteTqdm = ray.remote(tqdm_ray.tqdm)


@ray.remote
def process_array(array, perr, pbar):
    L = array.shape[0]
    n = L * L
    ray.get(pbar.update.remote(1))

    results = []

    if np.prod(array) == 1:
        results.append((0.25 + 0.25 / n) * TN(array, perr))

    for i in range(L):
        for j in range(L):
            flipped = array.copy()
            flipped[i, j] *= -1
            if np.prod(flipped) == 1:
                results.append((0.25 / n) * TN(flipped, perr))
            else:
                results.append(0.0)

    for i1 in range(L):
        for j1 in range(L):
            for i2 in range(L):
                for j2 in range(L):
                    if i1 == i2 and j1 == j2:
                        continue
                    flipped = array.copy()
                    flipped[i1, j1] *= -1
                    flipped[i2, j2] *= -1

                    if np.prod(flipped) == 1:
                        results.append((0.5 / (n**2)) * TN(flipped, perr))
                    else:
                        results.append(0.0)

    return np.sum(results)


for L in [5, 6, 7]:
    for perr in np.arange(0.0, 0.2, 0.01):
        measurements = (
            pd.read_csv(
                f"../traindata/toricpbcref3/d=2/measurements_L={L}_p={perr:.3f}.csv",
                delimiter=" ",
                header=None,
            )
            .dropna(axis=1, how="all")
            .to_numpy()
        )
        all_arrays = [np.array(m).reshape((L, L)) for m in measurements]
        total_tasks = len(all_arrays)
        pbar = RemoteTqdm.remote(total=total_tasks)
        futures = [process_array.remote(arr, perr, pbar) for arr in all_arrays]
        results = ray.get(futures)
        ray.get(pbar.close.remote())
        pd.DataFrame(results).to_csv(
            f"../traindata/toricpbcref3/d=2/ps_L={L}_p={perr:.3f}.csv",
            index=False,
            header=False,
        )
