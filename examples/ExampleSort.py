import Algorithm
import random
import ComplexityCalculator


class ExampleSort(Algorithm.Algorithm):

    def initial(self, n):
        x = list(range(n))
        random.shuffle(x)
        return x

    def calculate(self, structures):
        return sorted(structures)

    def garbage_collector(self):
        pass


algorithm = ExampleSort()
tc = ComplexityCalculator.approximate_complexity(algorithm, time_out=10, total_max_time=30, debug=True)