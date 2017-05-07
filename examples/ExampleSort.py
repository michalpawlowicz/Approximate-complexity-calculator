import random
import approximate_complexity_calculator as acc


class ExampleSort(acc.Algorithm):
    def __init__(self):
        pass

    def initial(self, n):
        x = list(range(n))
        random.shuffle(x)
        return x

    def calculate(self, structures):
        return sorted(structures)

    def garbage_collector(self):
        pass


algorithm = ExampleSort()
tc = acc.approximate_complexity(algorithm, time_out=10, total_max_time=30, debug=True)