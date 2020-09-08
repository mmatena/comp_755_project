"""Learns a simple policy using CMA."""
import cma

from absl import app
from absl import flags

import numpy as np


def fn(x):
    return np.sum((x - 5.0) ** 2)


es = cma.CMAEvolutionStrategy(8 * [0], 0.5)
for i in range(10):
    solutions = es.ask()
    fitlist = np.zeros(es.popsize)

    for i in range(es.popsize):
        fitlist[i] = fn(solutions[i])

    es.tell(fitlist)
    bestsol, bestfit = es.result()

# def main(_):
#     es = cma.CMAEvolutionStrategy(8 * [0], 0.5)
#     for i in range(10):
#         solutions = es.ask()
#         fitlist = np.zeros(es.popsize)

#         for i in range(es.popsize):
#             fitlist[i] = fn(solutions[i])

#         es.tell(fitlist)
#         bestsol, bestfit = es.result()


# if __name__ == "__main__":
#     app.run(main)

# help(cma.fmin)
# help(cma.CMAEvolutionStrategy)
# help(cma.CMAOptions)
