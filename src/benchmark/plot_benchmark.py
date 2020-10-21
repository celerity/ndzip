#!/usr/bin/env python3

# pipe input from benchmark binary into this script to plot throughput vs. compression ratio

import csv
import sys
from collections import defaultdict
from operator import itemgetter

import numpy as np
import scipy.stats as st
from matplotlib import patches, pyplot as plt
from tabulate import tabulate

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


class ThroughputStats:
    def __init__(self, dataset_points: list, op: str):
        samples = [np.array(
            [int(p['uncompressed bytes']) / float(t) * 1e6 for t in p[f'{op} times (microseconds)'].split(',')])
            for p in dataset_points]
        sample_means = [np.mean(ds) for ds in samples]
        self.mean = np.mean(sample_means)
        self.stddev = np.sqrt(np.mean([np.var(ds) for ds in samples]))
        self.min = np.mean([np.min(ds) for ds in samples])
        self.max = np.mean([np.max(ds) for ds in samples])
        # TODO is averaging error bar sizes correct?
        self.h95 = np.mean([st.t.ppf(1.95 / 2, len(ds) - 1) * st.sem(ds) for ds in samples])


if __name__ == '__main__':
    by_data_type_and_algorithm = defaultdict(lambda: defaultdict(list))
    algorithms = set()
    for f in input_files():
        rows = list(csv.reader(f, delimiter=';'))
        column_names = rows[0]
        for r in rows[1:]:
            a = dict(zip(column_names, r))
            by_data_type_and_algorithm[a['data type']][a['algorithm'], int(a['tunable'])].append(a)
            algorithms.add(a['algorithm'])

    data_type_means = []
    for row, data_type in enumerate(DATA_TYPES):
        means = []
        algo_means_dict = defaultdict(list)
        for (algo, tunable), points in by_data_type_and_algorithm[data_type].items():
            mean_compression_ratio = np.mean([float(a['compressed bytes']) / float(a['uncompressed bytes'])
                                              for a in points])
            throughput_stats = {op: ThroughputStats(points, op) for op in OPERATIONS}
            means.append((algo, tunable, mean_compression_ratio, throughput_stats))
            algo_means_dict[algo].append((mean_compression_ratio, throughput_stats))

        means.sort(key=itemgetter(0, 1))
        for v in algo_means_dict.values():
            v.sort(key=itemgetter(0))
        algo_means = sorted(algo_means_dict.items(), key=itemgetter(0))

        print(f'({data_type})')
        print(tabulate([[f'{a} {u}', '{:.3f}'.format(r).lstrip('0'),
                         *('{:,.0f} ± {:>3,.0f} MB/s'.format(t[o].mean * 1e-6, t[o].h95 * 1e-6) for o in OPERATIONS)]
                        for a, u, r, t in means], headers=['algorithm', 'ratio', *OPERATIONS], stralign='right',
                       disable_numparse=True))
        print()

        data_type_means.append((data_type, algo_means))

    fig, axes = plt.subplots(len(DATA_TYPES), len(OPERATIONS), figsize=(15, 10))
    fig.subplots_adjust(top=0.92, bottom=0.1, left=0.08, right=0.88, wspace=0.2, hspace=0.35)
    algorithm_colors = dict(zip(sorted(algorithms), PALETTE))
    for row, (data_type, algo_means) in enumerate(data_type_means):
        for col, operation in enumerate(OPERATIONS):
            ax = axes[row, col]
            throughput_values = []
            for algo, points in algo_means:
                throughputs, h95s, ratios = zip(*((t[operation].mean, t[operation].h95, r) for r, t in points))
                throughput_values += throughputs
                ax.errorbar(throughputs, ratios, label=algo, xerr=h95s,
                            color=algorithm_colors[algo], marker='D' if algo.startswith('new') else 'o')
            ax.set_title(f'{data_type} {operation}')
            ax.set_xscale('log')
            if throughput_values:
                ax.set_xlim(min(throughput_values) / 2, max(throughput_values) * 2)
            ax.set_xlabel('arithmetic mean uncompressed throughput [B/s]')
            ax.set_ylabel('arithmetic mean mean compression ratio')

    fig.legend(
        handles=[patches.Patch(color=c, label=a) for a, c in sorted(algorithm_colors.items(), key=itemgetter(0))],
        loc='center right')
    plt.show()
