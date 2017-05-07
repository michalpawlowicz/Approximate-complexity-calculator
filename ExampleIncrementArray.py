import Algorithm
import random
import ComplexityCalculator


class IncrementArray(Algorithm.Algorithm):

    def initial(self, n):
        x = list(range(n))
        random.shuffle(x)
        return x

    def calculate(self, structures):
        for i in structures:
            i += 1

    def garbage_collector(self):
        pass


algorithm = IncrementArray()
tc = ComplexityCalculator.approximate_complexity(algorithm, time_out=10, debug=True)
print(tc.predict_time(100000))