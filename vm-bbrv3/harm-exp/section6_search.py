"""Search strategies for the Reno vs. Westwood comparison in Section 6."""

from itertools import cycle, islice, product
from math import ceil, log2
from typing import NamedTuple


BANDWIDTHS = (25, 50, 100, 200, 400)  # Mbps
RTTS = (10, 20, 40, 80, 160, 320)  # ms
FLOW_COUNTS = (1, 2, 3, 4, 5)
QUEUE_LIMIT = 8192  # packets
BDP_MULTIPLES = (0.25, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5)
PACKET_BYTES = 1514


class Scenario(NamedTuple):
    bandwidth_mbps: int
    rtt_ms: int
    queue_packets: int
    alpha_flows: int
    beta_flows: int


def sweep_grid():
    # Reconstruct the packet queues in the original BESS sweep. The first
    # queue doubles with either bandwidth or RTT. Each link has up to five
    # successive powers of two, omitting queues beyond the BESS limit.
    return [
        Scenario(bw, rtt, queue, alpha, beta)
        for bw, rtt in product(BANDWIDTHS, RTTS)
        for queue in queue_sizes(bw, rtt)
        for alpha, beta in product(FLOW_COUNTS, repeat=2)
    ]


def queue_sizes(bw, rtt):
    first = 8 * (bw // BANDWIDTHS[0]) * (rtt // RTTS[0])
    return tuple(first << step for step in range(5) if first << step <= QUEUE_LIMIT)


def adjacent(axis, value):
    index = axis.index(value)
    return axis[max(0, index - 1):index] + axis[index + 1:index + 2]


def neighbors(scenario, dimension):
    bw, rtt, queue, alpha, beta = scenario
    queues = queue_sizes(bw, rtt)
    axes = (BANDWIDTHS, RTTS, queues, FLOW_COUNTS, FLOW_COUNTS)
    for value in adjacent(axes[dimension], scenario[dimension]):
        candidate = list(scenario)
        candidate[dimension] = value
        if dimension < 2:
            # Keep the same BDP-relative queue rank, or the closest rank
            # if the new link reaches the packet limit earlier.
            new_queues = queue_sizes(*candidate[:2])
            candidate[2] = new_queues[min(queues.index(queue), len(new_queues) - 1)]
        yield Scenario(*candidate)


def parameter_sweep(evaluate, grid, _rng):
    for scenario in grid[:evaluate.budget]:
        evaluate(scenario)


def random_search(evaluate, grid, rng):
    for scenario in rng.sample(grid, evaluate.budget):
        evaluate(scenario)


def hill_climbing(evaluate, grid, rng):
    restart_order = iter(rng.sample(grid, len(grid)))
    while evaluate.remaining:
        current = next(scenario for scenario in restart_order if scenario not in evaluate)
        current_harm = evaluate(current)
        while evaluate.remaining:
            improved = False
            # Generate each dimension's neighbors around the current point.
            # The next dimension sees any move made by the previous one.
            for dimension in range(len(Scenario._fields)):
                best = current
                for neighbor in neighbors(current, dimension):
                    if not evaluate.remaining:
                        return
                    harm = evaluate(neighbor)
                    if harm > current_harm:
                        best, current_harm = neighbor, harm
                if best != current:
                    current, improved = best, True
            if not improved:
                break


def log_uniform(rng, axis):
    return round(2 ** rng.uniform(log2(axis[0]), log2(axis[-1])))


def random_genes(rng):
    return (
        log_uniform(rng, BANDWIDTHS),
        log_uniform(rng, RTTS),
        rng.randrange(len(BDP_MULTIPLES)),
        rng.choice(FLOW_COUNTS),
        rng.choice(FLOW_COUNTS),
    )


def to_scenario(genes):
    bw, rtt, queue_index, alpha, beta = genes
    bdp_packets = bw * rtt * 1000 / (8 * PACKET_BYTES)
    target = BDP_MULTIPLES[queue_index] * bdp_packets
    queue = min(QUEUE_LIMIT, max(4, 2 ** ceil(log2(target))))
    return Scenario(bw, rtt, queue, alpha, beta)


def initial_population(size, rng):
    strata = (
        (0, BANDWIDTHS[0]), (0, BANDWIDTHS[-1]),
        (1, RTTS[0]), (1, RTTS[-1]),
        (2, 0), (2, len(BDP_MULTIPLES) - 1),
    )
    population = {}
    while len(population) < size:
        genes = list(random_genes(rng))
        gene, value = strata[len(population) % len(strata)]
        genes[gene] = value
        population.setdefault(to_scenario(genes), tuple(genes))
    return list(population.values())


def breed(first, second, rng):
    children = [list(first), list(second)]
    for index, (a, b) in enumerate(zip(first, second)):
        if rng.random() < 0.5:
            weight = rng.random()
            if index < 2:  # blend bandwidth and RTT in log space
                a, b = log2(a), log2(b)
            blended = (a + weight * (b - a), b + weight * (a - b))
            for child, value in zip(children, blended):
                child[index] = round(2 ** value if index < 2 else value)
    for child in children:
        if rng.random() < 0.3:
            for index, value in enumerate(random_genes(rng)):
                if rng.random() < 0.4:
                    child[index] = value
    return [tuple(child) for child in children]


def harmgen(evaluate, _grid, rng, population_size):
    population = initial_population(population_size, rng)
    while evaluate.remaining:
        # Distinct chromosomes can round to the same measured queue.
        candidates = {to_scenario(genes): genes for genes in population}
        scores = {}
        for scenario in candidates:
            if not evaluate.remaining:
                return
            scores[scenario] = evaluate(scenario)
        # Keep the top half for mating. If children collapse to duplicate
        # scenarios, breed more children from the distinct parents.
        ranked = sorted(scores, key=scores.get, reverse=True)
        parents = [candidates[scenario] for scenario in ranked[:population_size // 2]]
        rng.shuffle(parents)
        pairs = cycle(zip(parents[::2], parents[1::2]))
        children = [
            child
            for first, second in islice(pairs, ceil((population_size - len(parents)) / 2))
            for child in breed(first, second, rng)
        ]
        population = (parents + children)[:population_size]
