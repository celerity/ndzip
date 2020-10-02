#!/usr/bin/env python3

# pipe input from benchmark binary into this script to plot throughput vs. compression ratio

from collections import defaultdict
import matplotlib.pyplot as plt
import csv
import sys


def arithmetic_mean(x):
    return sum(x) / len(x)


def harmonic_mean(x):
    return 1 / (sum(1 / y for y in x) / len(x))


if len(sys.argv) > 1:
    with open(sys.argv[1], 'r') as f:
        rows = list(csv.reader(f, delimiter=';'))
else:
    rows = list(csv.reader(sys.stdin, delimiter=';'))

datasets = sorted(set(r[0] for r in rows[1:]))
dataset_idx = dict((d, i) for i, d in enumerate(datasets))
algos = sorted(set(r[3] for r in rows[1:]))
algos_idx = dict((a, i) for i, a in enumerate(algos))
table = [[''] + algos] + [['' for _ in range(len(algos) + 1)] for _ in range(len(datasets))]
for i, d in enumerate(datasets):
    table[i + 1][0] = d
for r in rows[1:]:
    table[1 + dataset_idx[r[0]]][1 + algos_idx[r[3]]] = str(int(r[6]) / int(r[5]))
for r in table:
    print(*r, sep=';')

benchmarks = defaultdict(lambda: defaultdict(list))
for r in rows[1:]:
    benchmarks[r[1]][r[3]].append((float(r[4]), int(r[5]), int(r[6])))

fig, axes = plt.subplots(len(benchmarks), 1)
throughput_values = []
for row, (data_type, algos) in enumerate(benchmarks.items()):
    ax = axes[row]
    for algorithm, points in algos.items():
        mean_compression_ratio = arithmetic_mean([c/u for t, u, c in points])
        mean_throughput = arithmetic_mean([u/t for t, u, c in points]) * 1e-6
        throughput_values.append(mean_throughput)
        ax.scatter(mean_throughput, mean_compression_ratio, label=algorithm)
    ax.set_title(data_type)
    ax.set_xscale('log')
    ax.set_xlim(min(throughput_values) / 2, max(throughput_values) * 2)
    ax.set_xlabel('average uncompressed throughput [MB/s]')
    ax.set_ylabel('harmonic mean compression ratio')
    ax.legend()
plt.show()
