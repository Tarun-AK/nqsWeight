import itertools
import math
from itertools import combinations

import pandas as pd


# 1) Helper to generate all set partitions of a list
def partitions(lst):
    if not lst:
        yield []
        return
    first = lst[0]
    for rest_part in partitions(lst[1:]):
        # Try inserting 'first' into each existing block
        for i in range(len(rest_part)):
            yield rest_part[:i] + [[first] + rest_part[i]] + rest_part[i + 1 :]
        # Or put 'first' as its own block
        yield [[first]] + rest_part


# Given a subset S (as a tuple of indices), yield all partitions of S
def get_partitions(S):
    if len(S) == 0:
        yield []
        return
    S_list = list(S)
    for part in partitions(S_list):
        # Each 'part' is a list of blocks; convert each block to a tuple
        yield [tuple(block) for block in part]


# 2) Generate all 2^5 = 32 configurations of 5 spins in {+1, -1}
n = 5
configs = list(itertools.product([-1, 1], repeat=n))

# 3) Compute H(σ) = ∏_{i=0..4} σ_i, then p(σ) = exp(H)/Z
H_vals = [math.prod(cfg) for cfg in configs]
exp_H = [math.exp(h) for h in H_vals]
Z = sum(exp_H)
probs = [val / Z for val in exp_H]


# 4) Compute all moments:
#    M_S = ⟨∏_{i∈S} σ_i⟩ = sum_{σ} [ (∏_{i∈S} σ_i) * p(σ) ]
def compute_moment(S, configs, probs):
    m = 0.0
    for cfg, p in zip(configs, probs):
        prod = 1
        for i in S:
            prod *= cfg[i]
        m += prod * p
    return m


moments = {}
for r in range(n + 1):
    for subset in combinations(range(n), r):
        moments[subset] = compute_moment(subset, configs, probs)


# 5) Compute cumulants via the partition‐sum formula:
#    κ_S = Σ_{P ⊢ S} (|P| - 1)! * (-1)^{|P|-1} * ∏_{B∈P} M_B
def compute_cumulant(S, moments):
    kappa = 0.0
    for part in get_partitions(S):
        num_blocks = len(part)
        sign = (-1) ** (num_blocks - 1)
        coeff = sign * math.factorial(num_blocks - 1)
        prod_m = 1.0
        for block in part:
            prod_m *= moments[block]
        kappa += coeff * prod_m
    return kappa


cumulants = {}
for r in range(n + 1):
    for subset in combinations(range(n), r):
        # By convention κ_∅ = 0
        if len(subset) == 0:
            cumulants[subset] = 0.0
        else:
            cumulants[subset] = compute_cumulant(subset, moments)

# 6) Display results in a table sorted by |S|, then lexicographically
rows = []
for subset in sorted(moments.keys(), key=lambda x: (len(x), x)):
    rows.append(
        {
            "Subset S": subset,
            "Moment M_S": moments[subset],
            "Cumulant κ_S": cumulants[subset],
        }
    )

df = pd.DataFrame(rows)
pd.set_option("display.float_format", "{:.8f}".format)
print(df)
