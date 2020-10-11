#!/usr/bin/env python3

# pipe input from benchmark binary into this script to plot throughput vs. compression ratio

from collections import defaultdict
from matplotlib import colors, patches, pyplot as plt
from operator import itemgetter
import math
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

column_names = rows[0]
data_types = ['float', 'double']
operations = ['compression', 'decompression']
palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
           '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5', '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5']

by_data_type_and_algorithm = defaultdict(lambda: defaultdict(list))
algorithms = set()
for r in rows[1:]:
    a = dict(zip(column_names, r))
    by_data_type_and_algorithm[a['data type']][a['algorithm']].append(a)
    algorithms.add(a['algorithm'])
algorithm_colors = dict(zip(sorted(algorithms), palette))

fig, axes = plt.subplots(len(data_types), len(operations))
fig.subplots_adjust(top=0.92, bottom=0.1, left=0.08, right=0.88, wspace=0.2, hspace=0.35)
for row, data_type in enumerate(data_types):
    for col, operation in enumerate(operations):
        ax = axes[row, col]
        throughput_values = []
        for algo, points in by_data_type_and_algorithm[data_type].items():
            mean_compression_ratio = arithmetic_mean(
                [float(a['compressed bytes']) / float(a['uncompressed bytes']) for a in points])
            mean_throughput = arithmetic_mean([float(a['uncompressed bytes'])
                                               / float(a[f'fastest {operation} time (seconds)'])
                                               for a in points]) * 1e-6
            throughput_values.append(mean_throughput)
            ax.scatter(mean_throughput, mean_compression_ratio, label=algo, color=algorithm_colors[algo],
                       marker='D' if algo.startswith('hcde') else 'o')
        ax.set_title(f'{data_type} {operation}')
        ax.set_xscale('log')
        if throughput_values:
            ax.set_xlim(min(throughput_values) / 2, max(throughput_values) * 2)
        ax.set_xlabel('arithmetic mean uncompressed throughput [MB/s]')
        ax.set_ylabel('arithmetic mean mean compression ratio')

fig.legend(handles=[patches.Patch(color=c, label=a) for a, c in sorted(algorithm_colors.items(), key=itemgetter(0))],
           loc='center right')
plt.show()
