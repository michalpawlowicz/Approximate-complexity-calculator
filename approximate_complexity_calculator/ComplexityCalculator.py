from .Algorithm import Algorithm
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
import logging
import datetime
from scipy.optimize import curve_fit
from enum import Enum


def sum_of_squares(x, y, fn):
    """
    Calculates error between training set and our approximation
    Residential sum of squares: sum(|f(x) - y|^2)
    for all (x,y) in out training set
    :return: float which represents error
    """
    return sum([(i - j) ** 2 for i, j in zip(map(fn, x), y)])


def nlogn(c, x):
    """
    :param c: coefficients array
    :param x: value
    :return: f(x)
    """
    return c[0] * x * np.log2(c[1] * x) + c[2]


def linear_logarithmic_regression(x, y, plot=False):
    """
    :param x: discrete set, with y creates training set
    :param y: discrete set, with x create training set
    :param plot: used for debugging, plot function and training set using matplotlib
    :return: array [a, b, c] which represents coefficients of function a * x * np.log2(b * x) + c
    """
    def fn(x, a, b, c):
        return a * x * np.log2(b * x) + c
    pars, pcov = curve_fit(
        f=fn,
        xdata=x,
        ydata=y,
        p0=[0.1, 0.1, 0.5],
        bounds=([0, 0, -np.inf],[np.inf, np.inf, np.inf])
    )
    if plot:
        x_plot = np.linspace(min(x), max(x), 100)
        y_plot = list(map(lambda i: fn(i, pars[0], pars[1], pars[2]), x_plot))
        plt.scatter(x, y)
        plt.plot(x_plot, y_plot)
        plt.xlabel("Problem size")
        plt.ylabel("Time")
    return pars


def polynomial_regression(x, y, n, plot=False):
    """
    Function does the same thing what linear_logarithmic_regression does
    :param n: Polynomial degree
    :return: array of coefficients
    """
    if n <= 0:
        raise ValueError("Regression degree must be greater then zero")
    if len(x) != len(y):
        raise ValueError("Arrays dimensions must be equal")
    z = np.polyfit(x, y, n)
    if plot:
        x_plot = np.linspace(min(x), max(x), 100)
        y_plot = list(map(lambda i: np.polyval(z, i), x_plot))
        plt.scatter(x, y)
        plt.plot(x_plot, y_plot)
        plt.xlabel("Problem size")
        plt.ylabel("Time")
    return z


def operation_time(seconds=5, logger=None):
    """
    Decorator, wraps function and meansure it's execution time, terminate it after time specified in timeout
    :param seconds: timeout in seconds, after that time function will be terminated
    :param logger: logger for errors
    :return: execution time of function in microseconds
    """
    def decorator(function):
        def wrapper(*args, **kwargs):
            p = multiprocessing.Process(target=function, args=args, kwargs=kwargs)
            start_time = datetime.datetime.now()
            p.start()
            p.join(seconds)
            if p.is_alive():
                p.terminate()
                p.join()
                if logger is not None:
                    logger.warn("Sample time out")
                return None
            else:
                finish_time = datetime.datetime.now()
                if logger is not None:
                    logger.info("Process finished successfully")
                return finish_time-start_time
        return wrapper
    return decorator


class BreakAllLoops(BaseException):
    pass


class ComplexityClasses(Enum):
    LINEAR = 1
    QUADRATIC = 2
    LINEAR_LOGARITHMIC = 3


class Complexity:
    def __init__(self, f_type: ComplexityClasses, coefficients, fn):
        self.f_type = f_type
        self.coefficients = coefficients
        self.fn = fn

    def predict_time(self, problem_size):
        return self.fn(self.coefficients, problem_size)

        # time in seconds
    def predict_size(self, time, left=1, right=25):
        if left + 1 == right:
            return 2 ** left;
        middle = int((left + right) / 2)
        if self.predict_time(2 ** middle) < time:
            return self.predict_size(time, middle, right)
        else:
            return self.predict_size(time, left, middle)


def approximate_complexity(algorithm : Algorithm, time_out=5, total_max_time=30, epsilon=0.1, debug=True):
    """
    :param algorithm: Object of class which implements Algorithm abstract class
    :param time_out: Time in seconds, max time how long one sample may take
    :param total_max_time: Time in seconds, max time how long whole approximation may take
    :param epsilon: helps to choose between linear function and square function, coefficient next to `x^2`
        may be almost numerical zero, so then `x^2` does not really matters.
        In that case linear approximation is a better option.
    :param debug: Plot result or not, if True logger writes to terminal
    :return: Object of class Complexity which represent time complexity.
    """
    if not isinstance(algorithm, Algorithm):
        raise ValueError("Class must extend Algorithm Class")
    if time_out <= 0:
        raise ValueError("Time out must be greater than zero")

    logger = logging.getLogger(__name__) # why __name__ ?
    # if variable debug is set to True, show logs in terminal
    if debug:
        logging.basicConfig(level=logging.DEBUG)

    @operation_time(seconds=time_out, logger=logger)
    def measure_time(problem_size):
        return algorithm.calculate(algorithm.initial(problem_size))

    def calculate_time():
        input_val = []
        output_val = []
        total_time = 0
        try:
            for i in range(2, 25):
                for j in range (0, 4):
                    problem_size = (2 ** i) + int(j/4 * (2 ** i))
                    if debug:
                        logger.info("Sampling problem size " + str(problem_size))
                    result = measure_time(problem_size)
                    if result is not None:
                        # None may be returned if sample require more computation time than time_out
                        input_val.append(problem_size)
                        output_val.append(result)
                        total_time += result.total_seconds()
                    else:
                        raise BreakAllLoops()

                    if total_time >= total_max_time:
                        # break from loops if total timeout was overstep
                        logger.warning("Total timeout")
                        raise BreakAllLoops()
        except BreakAllLoops:
            pass

        output_val = list(map(lambda x: x.total_seconds(), output_val))
        return input_val, output_val

    def choose_time_complexity():
        result = calculate_time()

        coe_quadratic = polynomial_regression(result[0], result[1], 2, debug)
        coe_linear = polynomial_regression(result[0], result[1], 1, debug)

        try:
            coe_lin_log = linear_logarithmic_regression(result[0], result[1], plot=debug)
            lin_log_error = sum_of_squares(
                result[0],
                result[1],
                lambda x: coe_lin_log[0]*x*np.log2(coe_lin_log[1]*x) + coe_lin_log[2]
            )
        except RuntimeError:
            logger.warning("Could not find optimal parameters for linear-logarithmic regression")
            coe_lin_log = None
            lin_log_error = None

        quadratic_error = sum_of_squares(result[0], result[1], lambda i: np.polyval(coe_quadratic, i))
        linear_error = sum_of_squares(result[0], result[1], lambda i: np.polyval(coe_linear, i))

        errors = [("Linear", linear_error), ("Quadratic", quadratic_error), ("Linear-logarithmic", lin_log_error)]
        if debug:
            logger.info("Value of residential sum of squares: " + str(errors))
        if debug:
            plt.show()

        c = min(errors, key=lambda x: x[1])
        if c[0] == "Linear-logarithmic":
            if debug:
                print("Approximate complexity: O(nlogn)")
            return Complexity(ComplexityClasses.LINEAR_LOGARITHMIC, coe_lin_log, nlogn)

        if c[0] == "Linear":
            if debug:
                print("Approximate complexity: O(N)")
            return Complexity(ComplexityClasses.LINEAR, coe_linear, np.polyval)

        if abs(quadratic_error - linear_error) < epsilon:
            if debug:
                print("Approximate complexity: O(N)")
            return Complexity(ComplexityClasses.LINEAR, coe_linear, np.polyval)
        else:
            if debug:
                print("Approximate complexity: O(N^2)")
            return Complexity(ComplexityClasses.QUADRATIC, coe_quadratic, np.polyval)

    return choose_time_complexity()
