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

algos = defaultdict(list)
for r in rows[1:]:
    algos[r[1]].append((float(r[2]), int(r[3]), int(r[4])))

fig, ax = plt.subplots(1, 1)
throughput_values = []
for algorithm, points in algos.items():
    mean_compression_ratio = harmonic_mean([c/u for t, u, c in points])
    mean_throughput = arithmetic_mean([u/t for t, u, c in points]) * 1e-6
    throughput_values.append(mean_throughput)
    ax.scatter(mean_throughput, mean_compression_ratio, label=algorithm)
ax.set_xscale('log')
ax.set_xlim(min(throughput_values) / 2, max(throughput_values) * 2)
ax.set_xlabel('average uncompressed throughput [MB/s]')
ax.set_ylabel('harmonic mean compression ratio')
plt.legend()
plt.show()
