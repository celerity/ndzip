#!/usr/bin/env python3

# pipe input from benchmark binary into this script to plot throughput vs. compression ratio

import csv
import sys
from collections import defaultdict
from operator import itemgetter

from matplotlib import patches, pyplot as plt

DATA_TYPES = ['float', 'double']
OPERATIONS = ['compression', 'decompression']
PALETTE = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
           '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5', '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5']


def arithmetic_mean(x):
    return sum(x) / len(x)


def input_files():
    if len(sys.argv) > 1:
        for n in sys.argv[1:]:
            with open(n, 'r') as f:
                yield f
    else:
        yield sys.stdin


if __name__ == '__main__':
    by_data_type_and_algorithm = defaultdict(lambda: defaultdict(list))
    algorithms = set()
    for f in input_files():
        rows = list(csv.reader(f, delimiter=';'))
        column_names = rows[0]
        for r in rows[1:]:
            a = dict(zip(column_names, r))
            by_data_type_and_algorithm[a['data type']][a['algorithm']].append(a)
            algorithms.add(a['algorithm'])

    fig, axes = plt.subplots(len(DATA_TYPES), len(OPERATIONS), figsize=(15, 10))
    fig.subplots_adjust(top=0.92, bottom=0.1, left=0.08, right=0.88, wspace=0.2, hspace=0.35)
    algorithm_colors = dict(zip(sorted(algorithms), PALETTE))
    for row, data_type in enumerate(DATA_TYPES):
        for col, operation in enumerate(OPERATIONS):
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

    fig.legend(
        handles=[patches.Patch(color=c, label=a) for a, c in sorted(algorithm_colors.items(), key=itemgetter(0))],
        loc='center right')
    plt.show()
