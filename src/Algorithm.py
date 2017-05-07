
from abc import ABC, abstractmethod

"""
Implementations of this class is used in complexity calculator,
all those methods are abstract so they must be oveloaded in extended classes
"""
from enum import Enum


class TimeComplexity(Enum):
    LINEAR = 1
    QUADRATIC = 2
    CUBIC = 3
    NLOGN = 4


class Algorithm(ABC):
    def __init__(self):
        pass
    # Takes as parameter N which is out problem size
    # for example N can be array size in some sorting algorithm
    # Result of this method must be passed to 'calculate' method
    @abstractmethod
    def initial(self, n):
        pass

    # 'calculate' method is a main algorithm.
    # Before using 'calculate' ones must to start initial method.
    # Uses data structures from initial method.
    @abstractmethod
    def calculate(self, structures):
        pass

    # Method should be called on the end
    # In general should work as garbage collector
    @abstractmethod
    def garbage_collector(self):
        pass