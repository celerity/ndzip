#!/usr/bin/env python3

from collections import defaultdict
from scipy.stats.mstats import hmean
import matplotlib.pyplot as plt
import csv
import sys

if len(sys.argv) > 1:
    with open(sys.argv[1], 'r') as f:
        rows = list(csv.reader(f, delimiter=';'))
else:
    rows = list(csv.reader(sys.stdin, delimiter=';'))

algos = defaultdict(list)
for r in rows[1:]:
    algos[r[1]].append((float(r[2]), float(r[3])))

points = [(sum(t for t, r in p), hmean([r for t, r in p]), a) for a, p in algos.items()]

fig, ax = plt.subplots(1, 1)
for time, ratio, algo in points:
    ax.scatter([time], [ratio], label=algo)
    print(time, ratio, algo)
ax.set_xscale('log')
ax.set_xlim(min(t for t, _, _ in points) / 2, max(t for t, _, _ in points) * 2)
ax.set_xlabel('time')
ax.set_ylabel('harmonic mean compression ratio')
plt.legend()
plt.show()
