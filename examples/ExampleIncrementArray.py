import approximate_complexity_calculator as acc
import random

class IncrementArray(acc.Algorithm):

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
tc = acc.approximate_complexity(algorithm, time_out=10, debug=True)
print(tc.predict_time(100000))
