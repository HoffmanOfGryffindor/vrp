from vrp import VRP
from glob import glob
import pandas as pd
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("problem", default=None)
    args = parser.parse_args()

    problems = glob("./data/*.txt")
    vrp = VRP(max_trips=10, cost_per_driver=500)

    if args.problem is None:
        problems = glob("./data/*.txt")
        for prob in problems:
            orders = pd.read_table(prob, delimiter=" ")
            orders = vrp.run(orders)
    else:
        orders = pd.read_table(args.problem, delimiter=" ")
        vrp = VRP(max_trips=len(orders), cost_per_driver=500)
        _ = [print(x) for x in vrp.run(orders)]
