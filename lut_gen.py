w = 32  # warp size / #banks
n = 16  # side length
q = w // n

def pad(i):
    return i + i // w - i // (n*n)

def d1(i):
    return i

def d2start(b):
    col = b % q * (n // q) + b // (q) + (b // n) * (n // q)
    return col % n + col//n * n*n

def d2(i):
    # b -> start mapping is LUT
    # pack 2x uint64_t indices so that a warp always reads only once from CM
    b = i // n
    start = d2start(b)
    return start + i % n * n

def d3start(b):
    return b // n + b % n * n

def d3(i):
    b = i // n
    start = d3start(b)
    return start + i % n * n * n

for d, s in (d1, 1), (d2, n), (d3, n*n):
    for i in range(n*n):
        b = 0
        banks = set()
        for j in range(n):
            a = b
            b = d(i * n + j)
            if j > 0:
                assert b - a == s
            k = pad(b) % w
            assert k not in banks
            banks.add(k)

print('const uint16_t lut2 = {', ', '.join(str(d2start(b)) for b in range(n ** 2)), '}')
print('const uint16_t lut3 = {', ', '.join(str(d3start(b)) for b in range(n ** 2)), '}')
