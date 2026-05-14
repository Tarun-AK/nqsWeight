import os
import pickle
import re
from collections import OrderedDict
from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import netket as nk
import numpy as np
import pandas as pd
from netket.experimental.models.fast_rnn import FastLSTMNet
from tqdm import tqdm

from models.conv import Conv2D
from models.lstm import FastLSTMNet as modLSTMNet
from models.mpdoRnn import MPDO
from models.resnet import ResidualConv2D
from models.transformer import Transformer

plt.style.use("~/plotStyle.mplstyle")


@partial(jax.vmap, in_axes=(0, None, None))
def prob(sigmas, L, p):
    size = L + (L % 2)
    minuses = jnp.array(jnp.argwhere(sigmas == -1, size=size, fill_value=jnp.nan))[:, 0]
    result = minuses[1::2] - minuses[::2]
    l = jnp.nansum(result)
    term1 = (p**l) * ((1 - p) ** (L - l))
    term2 = (p ** (L - l)) * ((1 - p) ** l)
    return jax.lax.cond(
        jnp.sum(sigmas == -1) % 2 == 0,
        lambda _: term1 + term2,
        lambda _: 0.0,
        operand=None,
    )


def reorder_serialization(new_array, L, d=2):
    N_samples, N = new_array.shape
    arr = new_array.reshape(N_samples, d, L, L)
    arr = arr.transpose(0, 2, 3, 1)
    return arr.reshape(N_samples, L * L * d)


def group(configs, L):
    num_samples = configs.shape[0]
    pairs = configs.reshape(num_samples, L * L, 2)
    idx = (pairs[..., 0] == -1).astype(int) * 2 + (pairs[..., 1] == -1).astype(int)
    mapping = np.array([-3, -1, 1, 3])
    return mapping[idx]


def inverse_transform(grouped_configs, L, D=2):
    n_samples = grouped_configs.shape[0]
    idx = (grouped_configs + 3) // 2

    x0_is_neg = (idx // 2).astype(int)
    x1_is_neg = (idx % 2).astype(int)

    x0 = 1 - 2 * x0_is_neg
    x1 = 1 - 2 * x1_is_neg

    mid_array = np.stack([x0, x1], axis=-1).reshape(n_samples, L, L, D)

    original_arr = mid_array.transpose(0, 3, 1, 2)

    return original_arr.reshape(n_samples, L * L * D)


def compute_syndromes(single_qubit, log_probability):
    L_target = single_qubit.shape[0]
    syndrome_list = []
    for i in range(0, L_target - 1):
        for j in range(0, L_target - 1):
            if (i + j) % 2 == 0:
                s_1 = single_qubit[i, j]
                s_2 = single_qubit[i + 1, j]
                s_3 = single_qubit[i, j + 1]
                s_4 = single_qubit[i + 1, j + 1]
                syndrome_list.append(s_1 * s_2 * s_3 * s_4)
    j = 0
    for i in range(1, L_target - 1, 2):
        s_1 = single_qubit[i, j]
        s_2 = single_qubit[i + 1, j]
        syndrome_list.append(s_1 * s_2)
    j = L_target - 1
    for i in range(0, L_target - 1, 2):
        s_1 = single_qubit[i, j]
        s_2 = single_qubit[i + 1, j]
        syndrome_list.append(s_1 * s_2)
    new_log_probability = log_probability + (L_target * L_target + 1) / 2
    return np.array(syndrome_list), new_log_probability


def build_graph_and_hilbert(L, ansatz, HType):

    if HType == "GHZ":
        g = nk.graph.Hypercube(length=L, n_dim=1, pbc=True)
        hilbert = nk.hilbert.Spin(s=0.5, N=g.n_nodes)
        return g, hilbert
    if "2D" in ansatz:
        g = nk.graph.Hypercube(length=L, n_dim=2, pbc=True)
    else:
        g = nk.graph.Hypercube(length=L**2, n_dim=1, pbc=True)
    hilbert = nk.hilbert.Spin(s=0.5, N=g.n_nodes)
    print("HILBERT IS WRONG!")
    if ("PQ" in HType) and ("Ungrouped" not in ansatz):
        hilbert = nk.hilbert.Spin(s=1.5, N=L**2)
        print("HILBERT IS RIGHT!")
    elif "Ungrouped" in ansatz:
        hilbert = nk.hilbert.Spin(s=0.5, N=61)
    return g, hilbert


def instantiate_model(ansatz, dh, nl, L, hilbert, g):
    if "RNN" in ansatz and "modLSTM" not in ansatz:
        model = FastLSTMNet(layers=3, features=dh, hilbert=hilbert, graph=g)
        sampler = nk.sampler.ARDirectSampler(hilbert=hilbert)
        prefac = 2
    elif "MPDO" in ansatz:
        model = MPDO(hilbert=hilbert, extra_hidden=dh)
        sampler = nk.sampler.ARDirectSampler(hilbert=hilbert)
        prefac = 1
    elif "modLSTM" in ansatz:
        model = modLSTMNet(layers=3, features=dh, hilbert=hilbert, graph=g)
        sampler = nk.sampler.ARDirectSampler(hilbert=hilbert)
        prefac = 2
    elif ansatz == "Product":
        model = FastLSTMNet(layers=2, features=5, hilbert=hilbert, graph=g)
        sampler = nk.sampler.ARDirectSampler(hilbert=hilbert)
        prefac = 2
    elif "ResidualConv2D" in ansatz:
        model = ResidualConv2D(
            layers=nl,
            features=dh,
            kernel_size=(
                3,
                5,
            ),
            hilbert=hilbert,
            L=L,
            W=L,
            channels=1,
            use_dropout=True,
            dropout_rate=0.0,
        )
        sampler = nk.sampler.ARDirectSampler(hilbert=hilbert)
        prefac = 2
    elif "Conv2D" in ansatz:
        model = nk.models.FastARNNConv2D(
            layers=nl, features=dh, kernel_size=(L // 2, L // 2), hilbert=hilbert
        )
        sampler = nk.sampler.ARDirectSampler(hilbert=hilbert)
        prefac = 2
    elif "Conv1D" in ansatz:
        model = nk.models.FastARNNConv1D(
            layers=nl, features=5, kernel_size=25, hilbert=hilbert
        )
        sampler = nk.sampler.ARDirectSampler(hilbert=hilbert)
        prefac = 2

    elif "Transformer" in ansatz:
        if "Big" in ansatz:
            num_layers = 2
            d_model = 24
            dff = 16
            num_heads = 4
        elif "Huge" in ansatz:
            num_layers = 4
            d_model = 48
            dff = 192
            num_heads = 4
        elif "Large" in ansatz:
            num_layers = 4
            d_model = 16
            dff = 48
            num_heads = 4
        else:
            num_layers = 2
            d_model = 12
            dff = 16
            num_heads = 2
        model = Transformer(
            hilbert=hilbert,
            num_layers=num_layers,
            num_heads=num_heads,
            d_model=d_model,
            dff=dff,
            autoreg=True,
        )
        sampler = nk.sampler.ARDirectSampler(hilbert=hilbert)
        prefac = 2
    else:
        model = FastLSTMNet(layers=2, features=5, hilbert=hilbert, graph=g)
        sampler = nk.sampler.ARDirectSampler(hilbert=hilbert)
        prefac = 2
    return model, sampler, prefac


def load_state_variables(statePath, copy, L, state, ansatz, HType, num_copies, new):
    if not new:
        path = statePath + f"_copy={copy}_L={L}"
        try:
            vars = nk.experimental.vqs.variables_from_file(
                filename=path, variables=state.variables
            )
        except Exception as e:
            print(e)
            return None
    else:
        linspace = np.arange(0, 1, 0.05) if HType == "GHZ" else np.arange(0, 0.2, 0.01)
        param = float(re.search(r"param=([0-9.]+)", statePath).group(1))
        idx = np.argwhere(np.isclose(linspace, param))[0][0]
        path = f"logs/{HType.lower()}/d=2/{ansatz}_L={L}.pkl"
        with open(path, "rb") as f:
            data = pickle.load(f)
        vars = jax.tree.map(lambda x: x[num_copies * idx + copy], data)
    return vars


def remote_compute(
    metric_type,
    measurements,
    logPs,
    ansatz,
    HType,
    L,
    dh,
    nl,
    statePath,
    copy,
    p=None,
    n_samples=256,
    Ns_training=100000,
    num_copies=-1,
    new=True,
):
    g, hilbert = build_graph_and_hilbert(L, ansatz, HType)
    model, sampler, prefac = instantiate_model(ansatz, dh, nl, L, hilbert, g)

    state = nk.vqs.MCState(sampler, model=model, n_samples=n_samples)
    _ = state.log_value(measurements[:1])  # Initialize with just 1 sample
    if metric_type == "KL":
        vars = load_state_variables(
            statePath, copy, L, state, ansatz, HType, num_copies, new
        )
        if vars is None:
            return np.nan
        state.variables = vars
        logPTheta = prefac * state.log_value(measurements)
        if logPTheta.shape != logPs.shape:
            return np.nan
        return float(np.mean(logPs - logPTheta))

    if metric_type == "JS":
        vars = load_state_variables(
            statePath, copy, L, state, ansatz, HType, num_copies, new
        )
        if vars is None:
            return np.nan
        state.variables = vars
        pSamples = measurements
        qSamples = state.sample()[0]

        if HType == "GHZ":
            P = lambda x: prob(x, L, p)

        Q = lambda x: jnp.exp(2 * state.log_value(x))
        M = lambda x: 0.5 * (P(x) + Q(x))
        a = jnp.mean(jnp.log(P(pSamples)) - jnp.log(M(pSamples)))
        b = jnp.mean(jnp.log(Q(qSamples)) - jnp.log(M(qSamples)))
        return float(0.5 * (a + b))

    if metric_type in ("training", "parity"):
        idx = np.random.randint(len(measurements), size=(Ns_training,))
        _ = measurements[idx]
        state = nk.vqs.MCState(sampler, model=model, n_samples=Ns_training)

        vars = load_state_variables(
            statePath, copy, L, state, ansatz, HType, num_copies, new
        )
        if vars is None:
            return np.nan
        state.variables = vars
        samples = state.sample()[0]
        return float(np.mean(jnp.prod(samples, axis=-1)))

    return np.nan


def load_measurements_and_logps(ansatz, data_dir, HType, L, d, p, data_mode="train"):

    if "ToricCoherent" in HType:
        filename = f"{data_mode}data/toriccoherent/L{L}.pkl"
        with open(filename, "rb") as f:
            etaDict = pickle.load(f)
        logPs = []
        measurements = []
        for i in range(len(etaDict[p])):
            measurement, logP = (
                compute_syndromes(etaDict[p][i]["sample"], etaDict[p][i]["logp"])
                if HType == "ToricCoherent"
                else (etaDict[p][i]["sample"].flatten(), etaDict[p][i]["logp"])
            )
            measurements.append(measurement)
            logPs.append(logP)
        return np.array(measurements), np.log(2) * np.array(logPs)[:, 0, 0]
    letter = "h" if "Ising" in HType else "p"
    delim = "\t" if "Ising" in HType else " "
    pname = f"{int(p)}" if "ToricCoherent" in HType else f"{p:.3f}"
    measurements = pd.read_csv(
        f"{data_mode}data/{HType.lower()}/d={d}/measurements_L={L}_{letter}={pname}.csv",
        delimiter=delim,
        header=None,
    )
    measurements = measurements.dropna(axis=1, how="all").to_numpy()
    if "PQ" in HType:
        measurements = reorder_serialization(measurements, L)
        if "Ungrouped" not in ansatz:
            measurements = group(measurements, L)
    if HType == "GHZ":
        logPs = np.log(prob(measurements, L, p) + 1e-20)
    else:
        ps = pd.read_csv(
            f"{data_mode}data/{HType.lower()}/d={d}/ps_L={L}_{letter}={pname}.csv",
            delimiter=delim,
            header=None,
        )
        ps = ps.to_numpy()
        logPs = np.log(ps)[:, 0]
        if HType == "ToricPBCRef":
            logPs = logPs - np.log(2)
    return measurements, logPs


def compute_metric_for_param_range(config, metric_type, plot=True):
    # NEW: Support for multiple data types
    data_types = config.get("data_types", None)
    if data_types is None:
        # Backward compatibility: single data type
        data_types = [config.get("data", "train")]

    d = config.get("d", 2)

    # NEW: Support for L-dependent configurations
    L_dependent_configs = config.get("L_dependent_configs", None)

    # OLD: Original flat list approach (still supported)
    dhs = config.get("dhs", [10, 15, 20, 25])
    nls = config.get("nls", [3, 5, 7])
    num_copies = config.get("num_copies")
    ansatz_list = config.get("ansatz_list", ["Conv2D_oct"])
    L_list = config.get("L_list", [4, 6, 8])

    HTypes_list = config.get("HTypes", ["ToricPQU"])
    pDict = config.get("pDict")
    labels = config.get("labels")
    colors = config.get("colors", ["red", "blue", "green", "orange", "yellow"])
    markers = config.get("markers", ["o", "s", "^", "d", "*"])

    # NEW: Support for data-type specific styling
    data_type_styles = config.get("data_type_styles", {})

    results = {}
    tasks = []
    task_metadata = []  # Store metadata for each task

    # Iterate over data types
    for data in data_types:
        # NEW: Use L-dependent configs if provided
        if L_dependent_configs is not None:
            for config_entry in L_dependent_configs:
                L = config_entry["L"]
                ansatz_input = config_entry["ansatz"]

                # Check for "new" flag per entry, fallback to global config default
                new_flag = config_entry.get("new", config.get("new", True))

                # Support list of ansatzes
                ansatz_list_local = (
                    ansatz_input if isinstance(ansatz_input, list) else [ansatz_input]
                )

                dh = config_entry.get("dh", 20)
                nl = config_entry.get("nl", 3)
                num_copy = config_entry.get("num_copies", 10)

                for HType in HTypes_list:
                    params = pDict[HType]

                    for p in params:
                        for ansatz in ansatz_list_local:
                            tasks.append(
                                (
                                    metric_type,
                                    data,
                                    HType,
                                    L,
                                    d,
                                    p,
                                    ansatz,
                                    dh,
                                    nl,
                                    num_copy,
                                    new_flag,
                                )
                            )
                            task_metadata.append(
                                {
                                    "ansatz_group": ansatz_input,
                                    "ansatz_individual": ansatz,
                                    "L": L,
                                    "HType": HType,
                                    "dh": dh,
                                    "nl": nl,
                                    "data": data,
                                    "p": p,
                                }
                            )

        # OLD: Original approach (still supported for backward compatibility)
        else:
            it1 = -1
            for ansatz in ansatz_list:
                it1 += 1
                dh = dhs[it1 % len(dhs)]
                nl = nls[it1 % len(nls)]
                num_copy = num_copies[it1 % len(num_copies)]
                new_flag = config.get("new", True)

                it2 = -1

                for L in L_list:
                    it2 += 1
                    for HType in HTypes_list:
                        params = pDict[HType]

                        for p in params:
                            tasks.append(
                                (
                                    metric_type,
                                    data,
                                    HType,
                                    L,
                                    d,
                                    p,
                                    ansatz,
                                    dh,
                                    nl,
                                    num_copy,
                                    new_flag,
                                )
                            )
                            task_metadata.append(
                                {
                                    "ansatz_group": ansatz,
                                    "ansatz_individual": ansatz,
                                    "L": L,
                                    "HType": HType,
                                    "dh": dh,
                                    "nl": nl,
                                    "data": data,
                                    "p": p,
                                }
                            )

    # fetch
    out = [remote_batch_metric(*task) for task in tqdm(tasks, desc="Computing metrics")]

    # Intermediate aggregation to find min across ansatz list
    # Key: (ansatz_group_identifier, HType, L, data, p) -> list of metrics
    temp_results = {}
    for (ansatz, HType, L, p, metric), metadata in zip(out, task_metadata):
        # Convert list to tuple for hashable key if it's a list
        ansatz_id = str(metadata["ansatz_group"])
        group_key = (ansatz_id, HType, L, metadata["data"], p)

        if group_key not in temp_results:
            temp_results[group_key] = []
        temp_results[group_key].append(metric)

    # Final reduction
    for (ansatz_id, HType, L, data, p), metrics in temp_results.items():
        # Find minimum (or max for parity) across the ansatz candidates for this p
        if metric_type == "parity":
            best_metric = np.nanmax(metrics)
        else:
            best_metric = np.nanmin(metrics)

        key = (ansatz_id, HType, L, data)
        if key not in results:
            # Reconstruct metadata for plotting (using group info)
            # Find any individual metadata from this group to preserve nl/dh/etc
            orig_meta = next(
                m
                for m in task_metadata
                if str(m["ansatz_group"]) == ansatz_id
                and m["L"] == L
                and m["data"] == data
            )
            results[key] = {"params": [], "metrics": [], "metadata": orig_meta}

        results[key]["params"].append(p)
        results[key]["metrics"].append(best_metric)

    # convert arrays and sort by p
    for key in results:
        ps = np.array(results[key]["params"])
        ms = np.array(results[key]["metrics"])
        sort_idx = np.argsort(ps)
        results[key]["params"] = ps[sort_idx]
        results[key]["metrics"] = ms[sort_idx]

    if plot:
        seen = set()
        it2 = -1

        for key in results:
            ansatz_id, HType, L, data = key
            ps = results[key]["params"]
            ms = results[key]["metrics"]
            metadata = results[key]["metadata"]
            nl = metadata["nl"]

            it2 += 1
            xvals = ps if "ToricCoherent" not in HType else (np.pi * ps) / 100

            # Determine styling based on data type
            if data in data_type_styles:
                style = data_type_styles[data]
                color = style.get("color", colors[it2 % len(colors)])
                marker = style.get("marker", markers[it2 % len(markers)])
                linestyle = style.get("linestyle", "-")
                alpha = style.get("alpha", 1.0)
            else:
                color = colors[it2 % len(colors)]
                marker = markers[it2 % len(markers)]
                linestyle = "-"
                alpha = 1.0

            # Use provided label or generate default
            if labels and it2 < len(labels):
                label = labels[it2]
            else:
                label = f"L={L}, {nl} layers, {data}"

            # Avoid duplicate labels
            if label in seen:
                label = None
            else:
                seen.add(label)

            plt.scatter(
                xvals,
                ms,
                marker=marker,
                color=color,
                edgecolor="black",
                label=label,
                alpha=alpha,
            )

        metricLabel = ""
        if metric_type == "KL":
            metricLabel = r"$D_{KL}(p||p_\theta)$"
        elif metric_type == "parity":
            metricLabel = r"$\expval{\prod_i^{N} \sigma_i}_{p_{\theta}}$"
        elif metric_type == "JS":
            metricLabel = r"$D_{JS}(p||p_\theta)$"

        plt.axvline(
            x=0.109, color="black", linestyle=":", linewidth=2, label="$p_c = 0.109$"
        )

        plt.title("PLACEHOLDER", color="white")
        plt.ylabel(metricLabel)
        plt.xlabel(r"$p_{err}$")
        plt.tight_layout()
        plt.legend()
        plt.savefig("images/metric_plot_2.png", bbox_inches="tight", dpi=500)
        plt.show()

    return results


def remote_batch_metric(
    metric_type, data, HType, L, d, p, ansatz, dh, nl, num_copies, new=True
):
    measurements, logPs = load_measurements_and_logps(
        ansatz, data, HType, L, d, p, data_mode=data
    )
    if measurements is None:
        return (ansatz, HType, L, p, np.nan)

    statePath = f"logs/{HType.lower()}/d={d}/{ansatz}_param={p:.3f}"
    start = 0 if new else 5
    # start = 0
    fut = [
        remote_compute(
            metric_type,
            measurements,
            logPs,
            ansatz,
            HType,
            L,
            dh,
            nl,
            statePath,
            copy,
            p,
            num_copies=num_copies,
            new=new,
        )
        for copy in tqdm(
            range(start, start + num_copies),
            desc=f"Copies for {ansatz} L={L} p={p:.3f}",
            leave=False,
        )
    ]
    vals = fut
    if metric_type == "parity":
        vals = np.nanmax(vals)
    else:
        vals = np.nanmin(vals)
    return (ansatz, HType, L, p, vals)


def compute_and_plot_kl_difference(
    config, HType_a, HType_b, ansatz, L, color, marker, label
):
    """
    Compute and plot KL difference between two HTypes.

    This computes the mean of the differences (rather than difference of means),
    i.e., mean(|KL_a - KL_b|) across all model copies, not |mean(KL_a) - mean(KL_b)|.

    Parameters:
    -----------
    ansatz : str or list
        Single ansatz string or list of ansatz strings. If a list is provided,
        differences will be computed for each ansatz variant, then averaged.
    """
    config_local = dict(config)
    config_local["HTypes"] = [HType_a, HType_b]

    # Handle both single ansatz and list of ansatzes
    ansatz_list = ansatz if isinstance(ansatz, list) else [ansatz]

    # NEW: Handle data types
    data = config_local.get("data_types", [config_local.get("data", "train")])[0]
    d = config_local.get("d", 2)

    # Get params from pDict
    HTypes_list = config_local.get("HTypes", [HType_a, HType_b])
    pDict = config_local.get("pDict")
    params = pDict[HType_a]  # Assume both HTypes have same params

    # Determine num_copies and other params from L_dependent_configs if available
    L_dependent_configs = config_local.get("L_dependent_configs", None)

    if L_dependent_configs is not None:
        # Find the config entry for this L
        config_entry = next((c for c in L_dependent_configs if c["L"] == L), None)
        if config_entry:
            dh = config_entry.get("dh", 20)
            nl = config_entry.get("nl", 3)
            num_copies = config_entry.get("num_copies", 10)
            new_flag = config_entry.get("new", config.get("new", True))
        else:
            # Fallback to defaults
            dh = config_local.get("dhs", [20])[0]
            nl = config_local.get("nls", [3])[0]
            num_copies = 10
            new_flag = config.get("new", True)
    else:
        # Use original flat approach
        dh = config_local.get("dhs", [20])[0]
        nl = config_local.get("nls", [3])[0]
        num_copies = config_local.get("num_copies", [10])
        if isinstance(num_copies, list):
            num_copies = num_copies[0]
        new_flag = config.get("new", True)

    # Storage for all individual KL values
    # Structure: {p_value: {"a": [kl1, kl2, ...], "b": [kl1, kl2, ...]}}
    kl_values = {p: {"a": [], "b": []} for p in params}

    # Load measurements and compute KL for each ansatz, each copy, each param
    for ansatz_individual in tqdm(ansatz_list, desc="Processing ansatzes"):
        for p in tqdm(
            params, desc=f"Computing KL for {ansatz_individual}", leave=False
        ):
            # Load measurements for both HTypes
            measurements_a, logPs_a = load_measurements_and_logps(
                ansatz_individual, data, HType_a, L, d, p, data_mode=data
            )
            measurements_b, logPs_b = load_measurements_and_logps(
                ansatz_individual, data, HType_b, L, d, p, data_mode=data
            )

            if measurements_a is None or measurements_b is None:
                continue

            statePath_a = (
                f"logs/{HType_a.lower()}/d={d}/{ansatz_individual}_param={p:.3f}"
            )
            statePath_b = (
                f"logs/{HType_b.lower()}/d={d}/{ansatz_individual}_param={p:.3f}"
            )

            start = 0 if new_flag else 5

            # Compute KL for each copy for HType_a
            tasks_a = [
                remote_compute(
                    "KL",
                    measurements_a,
                    logPs_a,
                    ansatz_individual,
                    HType_a,
                    L,
                    dh,
                    nl,
                    statePath_a,
                    copy,
                    p,
                    num_copies=num_copies,
                    new=new_flag,
                )
                for copy in tqdm(
                    range(start, start + num_copies),
                    desc=f"HType_a copies p={p:.3f}",
                    leave=False,
                )
            ]

            # Compute KL for each copy for HType_b
            tasks_b = [
                remote_compute(
                    "KL",
                    measurements_b,
                    logPs_b,
                    ansatz_individual,
                    HType_b,
                    L,
                    dh,
                    nl,
                    statePath_b,
                    copy,
                    p,
                    num_copies=num_copies,
                    new=new_flag,
                )
                for copy in tqdm(
                    range(start, start + num_copies),
                    desc=f"HType_b copies p={p:.3f}",
                    leave=False,
                )
            ]

            # Fetch results
            kls_a = tasks_a
            kls_b = tasks_b

            # Store individual KL values (filter out NaNs)
            kl_values[p]["a"].extend([kl for kl in kls_a if not np.isnan(kl)])
            kl_values[p]["b"].extend([kl for kl in kls_b if not np.isnan(kl)])

    # Now compute mean of differences for each param
    valid_params = []
    mean_diffs = []

    for p in params:
        kls_a = np.array(kl_values[p]["a"])
        kls_b = np.array(kl_values[p]["b"])

        if len(kls_a) == 0 or len(kls_b) == 0:
            continue

        # Match up pairs by taking minimum length
        # If you want all pairwise combinations, you could do that instead
        min_len = min(len(kls_a), len(kls_b))
        kls_a = kls_a[:min_len]
        kls_b = kls_b[:min_len]

        # Compute differences for each pair
        diffs = np.abs(kls_a - kls_b)

        # Take mean of differences
        mean_diff = np.mean(diffs)

        valid_params.append(p)
        mean_diffs.append(mean_diff)

    if not valid_params:
        print(f"Warning: Could not find results for ansatz={ansatz}, L={L}")
        return None

    params = np.array(valid_params)
    diffs = np.array(mean_diffs)

    plt.scatter(
        params, diffs, marker=marker, edgecolor="black", color=color, label=label
    )
    plt.axvline(
        x=0.109, color="black", linestyle=":", linewidth=2, label="$p_c = 0.109$"
    )
    plt.ylabel(r"$\Delta S$")
    plt.xlabel(r"$p_{err}$")
    plt.tight_layout()

    return params, diffs


if __name__ == "__main__":
    L = 64

    # EXAMPLE: PLOTTING BOTH TRAIN AND TEST DATA
    config = {
        # NEW: Specify multiple data types to plot
        "data_types": ["test"],
        # # NEW: Optional styling for different data types
        # "data_type_styles": {
        #      "train": {"color": "blue", "marker": "o", "alpha": 0.7},
        #      "test": {"color": "red", "marker": "s", "alpha": 0.7},
        # },
        "nls": [3],
        "dhs": [20],
        "d": 2,
        "L_dependent_configs": [
            {
                "L": 4,
                "ansatz": [
                    "ResidualConv2D_nl=2_dh=24_dp=0.2_oct",
                    "ResidualConv2D_nl=2_dh=24_dp=0.2_42_oct",
                ],
                "nl": 2,
                "dh": 24,
                "num_copies": 5,
            },
            {
                "L": 6,
                "ansatz": [
                    "ResidualConv2D_nl=3_dh=24_dp=0.2_oct",
                    "ResidualConv2D_nl=3_dh=24_dp=0.2_42_oct",
                    "ResidualConv2D_nl=3_dh=24_dp=0.2_111_oct",
                    "ResidualConv2D_nl=3_dh=24_dp=0.2_1111_oct",
                    "ResidualConv2D_nl=3_dh=24_dp=0.2_422_oct",
                    "ResidualConv2D_nl=3_dh=24_dp=0.2_4222_oct",
                    "ResidualConv2D_nl=3_dh=24_dp=0.2_11111_oct",
                ],
                "nl": 3,
                "dh": 24,
                "num_copies": 5,
            },
            {
                "L": 8,
                "ansatz": [
                    "ResidualConv2D_nl=4_dh=24_dp=0.2_oct",
                    "ResidualConv2D_nl=4_dh=24_dp=0.2_42_oct",
                    "ResidualConv2D_nl=4_dh=24_dp=0.2_111_oct",
                    "ResidualConv2D_nl=4_dh=24_dp=0.2_1111_oct",
                    "ResidualConv2D_nl=4_dh=24_dp=0.2_422_oct",
                    "ResidualConv2D_nl=4_dh=24_dp=0.2_4222_oct",
                    "ResidualConv2D_nl=4_dh=24_dp=0.2_11111_oct",
                ],
                "nl": 4,
                "dh": 24,
                "num_copies": 5,
            },
            # {
            #     "L": 7,
            #     "ansatz": [
            #         "ResidualConv2D_nl=1_dh=24_dp=0.2_oct",
            #     ],
            #     "nl": 1,
            #     "dh": 24,
            #     "num_copies": 5,
            #     "new": True,
            # },
            # {
            #     "L": 7,
            #     "ansatz": [
            #         "ResidualConv2D_nl=2_dh=24_dp=0.2_oct",
            #     ],
            #     "nl": 2,
            #     "dh": 24,
            #     "num_copies": 5,
            #     "new": True,
            # },
            # {
            #     "L": 7,
            #     "ansatz": [
            #         "ResidualConv2D_nl=3_dh=24_dp=0.2_oct",
            #     ],
            #     "nl": 3,
            #     "dh": 24,
            #     "num_copies": 5,
            #     "new": True,
            # },
            # {
            #     "L": 7,
            #     "ansatz": [
            #         "TransformerBig_oct",
            #     ],
            #     "nl": 1,
            #     "dh": 24,
            #     "num_copies": 5,
            #     "new": True,
            # },
            # {
            #     "L": 64,
            #     "ansatz": [
            #         "MPDO_oct",
            #     ],
            #     "nl": 3,
            #     "dh": 10,
            #     "num_copies": 5,
            #     "new": True,
            # },
        ],
        "labels": [
            "L=4, nl=2",
            "L=6, nl=3",
            "L=8, nl=4",
            # "Transformer",
            # "RNN (initialized to parity)",
            # "1 Block",
            # "2 Blocks",
            # "3 Blocks",
        ],
        "HTypes": ["ToricOpenPQUV3"],
        "pDict": {
            "Toric": np.arange(0.00, 0.2, 0.01),
            "ToricPBC": np.arange(0.01, 0.2, 0.01),
            "ToricPBCRef": np.arange(0.0, 0.2, 0.01),
            "ToricPBCRef2": np.arange(0.0, 0.2, 0.01),
            "ToricPBCRef4": np.arange(0.0, 0.2, 0.01),
            "ToricPQU": np.arange(0.0, 0.2, 0.01),
            "SToricPQU": np.arange(0.0, 0.2, 0.01),
            "ToricOpenPQU": np.arange(0.0, 0.2, 0.01),
            "ToricOpenPQUV2": np.arange(0.0, 0.2, 0.01),
            "ToricOpenPQUV3": np.arange(0.0, 0.2, 0.01),
            "ToricOpenPQV3": np.arange(0.0, 0.2, 0.01),
            "ToricPQ": np.arange(0.0, 0.2, 0.01),
            "IsingX": np.arange(0.05, 2, 0.05),
            "XXpZ": np.arange(0.05, 2, 0.05),
            "IsingZ": np.arange(0.05, 2, 0.05),
            "GHZ": np.arange(0.0, 1.0, 0.05),
            "ToricCoherent": np.arange(0, 26),
            "ToricCoherentSQ": np.arange(0, 26),
            "LongRange": [0.0],
            "ShortRange": [0.0],
        },
    }

    # This will now plot both train and test data
    results = compute_metric_for_param_range(config, metric_type="KL")

    # plt.figure(figsize=(10, 7))
    # compute_and_plot_kl_difference(
    #     config, "ToricOpenPQUV3", "ToricOpenPQV3", "Conv2D_oct", 4, "red", "o"
    # )
    # compute_and_plot_kl_difference(
    #     config, "ToricOpenPQUV3", "ToricOpenPQV3", "Conv2D_oct", 6, "blue", "s"
    # )
    # compute_and_plot_kl_difference(
    #     config,
    #     "ToricOpenPQUV3",
    #     "ToricOpenPQV3",
    #     ["Conv2D_oct"],
    #     8,
    #     "red",
    #     "o",
    #     label="10 inits",
    # )
    # compute_and_plot_kl_difference(
    #     config,
    #     "ToricOpenPQUV3",
    #     "ToricOpenPQV3",
    #     ["Conv2D_oct", "Conv2D_422_oct", "Conv2D_1111_oct"],
    #     8,
    #     "green",
    #     "^",
    #     label="30 inits",
    # )
    # plt.legend()
    # plt.savefig("images/kl_difference", dpi=500)
