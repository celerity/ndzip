#!/usr/bin/env python3

# pipe input from benchmark binary into this script to plot throughput vs. compression ratio

import csv
import sys
from collections import defaultdict
from operator import itemgetter
from argparse import ArgumentParser
from math import floor, ceil, log10

import numpy as np
import scipy.stats as st
from matplotlib import patches, ticker, pyplot as plt
from tabulate import tabulate

DATA_TYPES = ['float', 'double']
OPERATIONS = ['compression', 'decompression']
PALETTE = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
           '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5', '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5']


def arithmetic_mean(x):
    return sum(x) / len(x)


def input_files(file_list):
    if file_list:
        for n in file_list:
            if n == '-':
                yield sys.stdin
            else:
                with open(n, 'r') as f:
                    yield f
    else:
        yield sys.stdin


class ThroughputStats:
    def __init__(self, dataset_points: list, op: str):
        sample_means = [int(p['uncompressed bytes']) / np.mean(np.array(
            [float(t) for t in p[f'{op} times (microseconds)'].split(',')])) * 1e6
                        for p in dataset_points]
        # TODO stats except mean are probably imprecise
        samples = [np.array(
            [int(p['uncompressed bytes']) / float(t) * 1e6 for t in p[f'{op} times (microseconds)'].split(',')])
            for p in dataset_points]
        self.mean = np.mean(sample_means)
        self.stddev = np.sqrt(np.mean([np.var(ds) for ds in samples]))
        self.min = np.mean([np.min(ds) for ds in samples])
        self.max = np.mean([np.max(ds) for ds in samples])
        # TODO is averaging error bar sizes correct?
        self.h95 = np.mean([st.t.ppf(1.95 / 2, len(ds) - 1) * st.sem(ds) for ds in samples])


def log_ticks(start: float, stop: float, step: int):
    ticks = []
    base = 10 ** floor(log10(start))
    mul = ceil(start / base)
    while mul * base <= stop:
        ticks.append(mul * base)
        mul += step
        if mul >= 10:
            base *= 10
            mul = 1
    return ticks


def plot_throughput_vs_ratio(algorithms, by_data_type_and_algorithm, output_pgf):
    data_type_means = []
    for row, data_type in enumerate(DATA_TYPES):
        means = []
        algo_means_dict = defaultdict(list)
        for algo, results_by_tunable in by_data_type_and_algorithm[data_type].items():
            for tunable, results_by_num_threads in results_by_tunable.items():
                max_threads_results = results_by_num_threads[max(results_by_num_threads.keys())]
                mean_compression_ratio = np.mean([float(a['compressed bytes']) / float(a['uncompressed bytes'])
                                                  for a in max_threads_results])
                throughput_stats = {op: ThroughputStats(max_threads_results, op) for op in OPERATIONS}
                means.append((algo, tunable, mean_compression_ratio, throughput_stats))
                algo_means_dict[algo].append((tunable, mean_compression_ratio, throughput_stats))

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

    fig, axes = plt.subplots(len(DATA_TYPES), len(OPERATIONS), figsize=(10, 6))
    fig.subplots_adjust(top=0.92, bottom=0.1, left=0.08, right=0.88, wspace=0.2, hspace=0.35)
    algorithm_colors = dict(zip(sorted(algorithms), PALETTE))
    for row, (data_type, algo_means) in enumerate(data_type_means):
        for col, operation in enumerate(OPERATIONS):
            ax = axes[row, col]
            throughput_values = []
            for algo, results in algo_means:
                points = [(t[operation].mean, t[operation].h95, r) for _, r, t in results]
                points.sort(key=itemgetter(0))
                throughputs, h95s, ratios = zip(*points)
                throughput_values += throughputs
                if len(points) > 1:
                    marker = None
                elif algo.startswith('ndzip'):
                    marker = 'D'
                else:
                    marker = 'o'
                ax.errorbar(throughputs, ratios, label=algo, xerr=h95s, color=algorithm_colors[algo], marker=marker,
                            linewidth=2)
            ax.set_title(f'{data_type} {operation}')
            ax.set_xscale('log')
            if throughput_values:
                ax.set_xlim(min(throughput_values) / 2, max(throughput_values) * 2)
            ax.set_xlabel('arithmetic mean uncompressed throughput [B/s]')
            ax.set_ylabel('arithmetic mean compression ratio')

    fig.legend(
        handles=[patches.Patch(color=c, label=a) for a, c in sorted(algorithm_colors.items(), key=itemgetter(0))],
        loc='center right')

    if output_pgf:
        plt.savefig('benchmark.pgf')
    else:
        plt.show()


def plot_scaling(algorithms, by_data_type_and_algorithm, output_pgf):
    data_type_means = []
    for row, data_type in enumerate(DATA_TYPES):
        means = []
        algo_means_dict = defaultdict(list)
        for algo, results_by_tunable in by_data_type_and_algorithm[data_type].items():
            max_tunable_results = results_by_tunable[max(results_by_tunable.keys())]
            if len(max_tunable_results) > 1:
                for num_threads, results in max_tunable_results.items():
                    throughput_stats = {op: ThroughputStats(results, op) for op in OPERATIONS}
                    means.append((algo, num_threads, throughput_stats))
                    algo_means_dict[algo].append((num_threads, throughput_stats))

        means.sort(key=itemgetter(0, 1))
        for v in algo_means_dict.values():
            v.sort(key=itemgetter(0))
        algo_means = sorted(algo_means_dict.items(), key=itemgetter(0))

        print(f'({data_type})')
        print(tabulate([[f'{a} {u}',
                         *('{:,.0f} ± {:>3,.0f} MB/s'.format(t[o].mean * 1e-6, t[o].h95 * 1e-6) for o in OPERATIONS)]
                        for a, u, t in means], headers=['algorithm', *OPERATIONS], stralign='right',
                       disable_numparse=True))
        print()

        data_type_means.append((data_type, algo_means))

    fig, axes = plt.subplots(len(DATA_TYPES), len(OPERATIONS), figsize=(10, 6))
    fig.subplots_adjust(top=0.92, bottom=0.1, left=0.08, right=0.88, wspace=0.2, hspace=0.35)
    algorithm_colors = dict(zip(sorted(algorithms), PALETTE))
    for row, (data_type, algo_means) in enumerate(data_type_means):
        for col, operation in enumerate(OPERATIONS):
            ax = axes[row, col]
            throughput_values = []
            for algo, results in algo_means:
                points = [(threads, t[operation].mean, t[operation].h95) for threads, t in results]
                threads, throughputs, h95s = zip(*points)
                throughput_values += throughputs
                ax.errorbar(threads, throughputs, label=f'{algo} {data_type} {operation}', yerr=h95s, marker='o')
            ax.set_title(f'{data_type} {operation}')
            ax.set_xscale('log')
            ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
            ax.xaxis.set_minor_formatter(ticker.ScalarFormatter())
            ax.set_yscale('log')
            if throughput_values:
                start, stop = min(throughput_values) / 2, max(throughput_values) * 2
                ax.set_ylim(start, stop)
                ax.yaxis.set_minor_formatter(ticker.LogFormatterSciNotation(minor_thresholds=(2, 0.5)))
            ax.set_xlabel('number of threads')
            ax.set_ylabel('arithmetic mean uncompressed throughput [B/s]')

    fig.legend(
        handles=[patches.Patch(color=c, label=a) for a, c in sorted(algorithm_colors.items(), key=itemgetter(0))],
        loc='center right')

    if output_pgf:
        plt.savefig('scaling.pgf')
    else:
        plt.show()


def main():
    parser = ArgumentParser(description='Visualize benchmark results')
    parser.add_argument('csv_files', metavar='CSVS', nargs='*', help='benchmark csv files')
    parser.add_argument('--scaling', action='store_true', help='plot scaling (default: throughput vs ratio)')
    parser.add_argument('--pgf', action='store_true', help='output pgfplots')
    args = parser.parse_args()

    by_data_type_and_algorithm = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    algorithms = set()
    for f in input_files(args.csv_files):
        rows = list(csv.reader(f, delimiter=';'))
        column_names = rows[0]
        for r in rows[1:]:
            a = dict(zip(column_names, r))
            num_threads = int(a.get('number of threads', 1))
            by_data_type_and_algorithm[a['data type']][a['algorithm']][int(a['tunable'])][num_threads].append(a)
            algorithms.add(a['algorithm'])

    if not args.scaling:
        plot_throughput_vs_ratio(algorithms, by_data_type_and_algorithm, args.pgf)
    else:
        plot_scaling(algorithms, by_data_type_and_algorithm, args.pgf)


if __name__ == '__main__':
    main()
