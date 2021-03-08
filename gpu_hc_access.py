"""
Simulates shared-memory access through gpu::hypercube_layout and gpu::directional_accessor and reports bank conflicts
for the semantics of CUDA architectures >= 5.x
"""

from collections import namedtuple

Layout = namedtuple('Layout', ['dimensions', 'num_lanes', 'pad', 'accessors'])
Accessor = namedtuple('Accessor', ['direction', 'offset', 'stride'])

WARP_SIZE = 32
HC_SIZE = 4096
LAYOUTS = [
    Layout(
        dimensions=1,
        num_lanes=256,  # Only for forward transform - 256 lanes in series
        pad=lambda i, w: w * i + w * i // 32,
        accessors=[
            Accessor(
                direction='H',
                offset=lambda tid: tid * HC_SIZE // 256,
                stride=1,
            ),
        ]
    ),
    Layout(
        dimensions=2,
        num_lanes=256,  # Only for forward transform (4 lanes in series per row/col)
        pad=lambda i, w: w*i + w*i // 32,
        accessors=[
            Accessor(
                direction='H',
                offset=lambda tid: tid * (HC_SIZE // 256),
                stride=1,
            ),
            Accessor(
                direction='V',
                offset=lambda tid: (tid // 64) * (HC_SIZE // 256 * 64) + tid % 64,
                stride=64,
            ),
        ]
    ),
    Layout(
        dimensions=2,
        num_lanes=64,  # For non-scanning inverse transform
        # 32-bit: pad once every row
        # 64-bit: pad every 16 values, but not at the end of each line - resulting bank-stride of 3 is conflict free!
        pad=lambda i, w: w * i + (i // 64 if w == 1 else i // 16 - i // 64),
        accessors=[
            Accessor(
                direction='H',
                offset=lambda tid: tid * 64,
                stride=1,
            ),
            Accessor(
                direction='V',
                offset=lambda tid: tid % 64,
                stride=64,
            ),
        ]
    ),
    Layout(
        dimensions=3,
        num_lanes=256,  # For both forward and inverse transform
        pad=lambda i, w: w * i + w * i // 32 - (i // 512 if w == 2 else 0),
        accessors=[
            Accessor(
                direction='H',
                offset=lambda tid: tid * 16,
                stride=1,
            ),
            Accessor(
                direction='V',
                offset=lambda tid: (tid // 16) * 512 - (tid // 128) * (HC_SIZE - 256) + tid % 16,
                stride=16,
            ),
            Accessor(
                direction='P',
                offset=lambda tid: tid,
                stride=256,
            ),
        ]
    ),
]


for layout in LAYOUTS:
    lane_length = HC_SIZE // layout.num_lanes
    for value_width in 1, 2:
        for acc in layout.accessors:
            cell_accesses = [False for _ in range(HC_SIZE)]
            for step in range(lane_length):
                for warp in range(layout.num_lanes // WARP_SIZE):
                    for word in range(value_width):
                        bank_accesses = [[] for _ in range(WARP_SIZE)]
                        for warp_tid in range(WARP_SIZE):
                            tid = warp * WARP_SIZE + warp_tid
                            index = acc.offset(tid) + acc.stride * step
                            cell_accesses[index] = True
                            bank = (layout.pad(index, value_width) + word) % WARP_SIZE
                            bank_accesses[bank].append((tid, index))
                        for bank, conflict in enumerate(bank_accesses):
                            if len(conflict) > 1:
                                print('Bank conflict:',
                                      f'{layout.dimensions}Dx{layout.num_lanes}/{acc.direction} step {step}',
                                      f'word offset {word}:' if value_width > 1 else '',
                                      f'Bank {bank:2} accessed by threads',
                                      ', '.join(f'{tid:3} ({index:4} padded {layout.pad(index, value_width) + word:4})'
                                                for tid, index in conflict))
            for cell, accessed in enumerate(cell_accesses):
                if not accessed:
                    print(f'Cell not accessed: {cell} in {layout.dimensions}D/{acc.direction}')
