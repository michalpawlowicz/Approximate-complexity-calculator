# Approximate-complexity-calculator

## Installation

## Example
### Bubble-Sort
First thing to do is implementation of `Algorithm` class.<br />
Complexity Calculator only takes as parameter subclasses of Algorithm abstract class.<br />
Meaning of thoes three function - see more in documentation.<br />
```python
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
```

```python
algorithm = ExampleSort()tc = approximate_complexity_calculator.approximate_complexity(algorithm, time_out=5, debug=True)
print(tc.predict_size(4)) # returns size of array which can be soreted in less than 4 seconds
print(tc.predict_time(1024)) # requaire time for sorting array of 1024 elements
```

## Documentation 
`documentation` folder contain full documentation in html format.
[link](https://github.com/michalpawlowicz/Approximate-complexity-calculator/tree/master/documentation)

## Requirements
* Python 3.6
