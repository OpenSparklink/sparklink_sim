"""Independent source-address dictionary oracle;validation is opened only after freeze."""

from random import Random

MAX_START = (1 << 64) - 5


def prefix_oracle(previous_start, next_start):
    for value in (previous_start, next_start):
        if type(value) is not int:
            raise ValueError("Native address required")
        if value < 0 or value > MAX_START:
            raise OverflowError("Four-word source window outside64bit exclusive end")
    source = {previous_start + i: i for i in range(4)}
    count = 0
    for address in range(next_start, next_start + 4):
        if address not in source:
            break
        count += 1
    return count


def address_vectors(split):
    if split == "development":
        bases = (0, (1 << 16) - 7, (1 << 32) + 23, (1 << 63) - 11)
        shifts = (-6, -1, 0, 1, 2, 3, 4, 7)
        random, count = Random(2026112011), 64
        bits = 63
    elif split == "validation":
        bases = (1, (1 << 24) + 9, (1 << 63) + 17, MAX_START)
        shifts = (-(1 << 40), -5, 0, 1, 2, 3, 4, 1 << 40)
        random, count = Random(2026112037), 80
        bits = 64
    else:
        raise ValueError("Unknown address-node split")
    for base in bases:
        for shift in shifts:
            yield base, base + shift
    for _ in range(count):
        base = random.randrange(1 << bits)
        yield base, base + random.choice(shifts)
