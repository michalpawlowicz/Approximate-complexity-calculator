import random
import approximate_complexity_calculator


class ExampleBubbleSort(approximate_complexity_calculator.Algorithm):
    def initial(self, n):
        x = list(range(n))
        random.shuffle(x)
        return x

    def calculate(self, structures):
        def bubble_sort(alist):
            for passnum in range(len(alist) - 1, 0, -1):
                for i in range(passnum):
                    if alist[i] > alist[i + 1]:
                        temp = alist[i]
                        alist[i] = alist[i + 1]
                        alist[i + 1] = temp
        bubble_sort(structures)

    def garbage_collector(self):
        pass


algorithm = ExampleBubbleSort()
tc = approximate_complexity_calculator.approximate_complexity(algorithm, time_out=5, debug=True)
print(tc.predict_size(3))
