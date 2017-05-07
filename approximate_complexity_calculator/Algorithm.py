from abc import ABC, abstractmethod


class Algorithm(ABC):
    """
    Class prepared for extending.
    ComplexityCalculator uses only classes which inherit from this abstract class.
    """
    def __init__(self):
        pass

    @abstractmethod
    def initial(self, n):
        """
        Takes as parameter N, which is out problem size.
        For example, N can be array size in some sorting algorithm.
        Result of this method must be passed to 'calculate' method
        """
        pass

    @abstractmethod
    def calculate(self, structures):
        """
        'calculate' method is a main algorithm.
        Before using 'calculate' ones must to start initial method.
        Uses data structures from initial method.
        It is not important what calculate method returns.
        """
        pass

    @abstractmethod
    def garbage_collector(self):
        """
        Method should be called on the end
        In general should work as garbage collector
        """
        pass
