import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import random
import multiprocessing
from functools import lru_cache


class Ant:
    """This class represents our clueless ant in search for food! Initialize the ant passing the following parameters:

    Args:
        boundary_function (function): this is a function that takes as input two integers x and y, representig a point in Z^2. It evaluates if the given (x,y) point is inside the close boundary and return True if it is. Otherwise, it returns False.
        x0 (int): starting x coordinate of the ant. Defaults to 0.
        y0 (int): starting y coordinate of the ant. Defaults to 0.
        workers (int): if workers is passed, the monte carlo simulation is done using parallelisation with multiprocessing, to speed up calculations. If None, no parallelisation is used. Defaults to None.
    """

    def __init__(self,
                 boundary_function,
                 x0=0,
                 y0=0,
                 workers=None) -> None:

        self.x0 = x0
        self.y0 = y0
        self.workers = workers
        self.eval_bound = self._make_njitted_function(boundary_function)

        print('Hi! My name is Bill, and I am a clueless ant is search for food!')
        print(
            f'In my search for food, I will start from ({self.x0}, {self.y0})')
        print('Please wait while I calculate all the transition points that I can visit (even I should not be aware of them xD)')
        self.points = self._create_admissible_points()
        print('Ok, I am ready to go!')

    """ Static Methods """
    @staticmethod
    def _make_njitted_function(f) -> object:
        """Just a wrapper to jit the passed function

        Args:
            f (function): The function to jit

        Returns:
            funtion: jitted funtion
        """
        return njit(f)

    @staticmethod
    @njit
    def _get_points(eval_boundary, size) -> list:
        """This is used to retrieve all points that satisfy the boundary condition, given a jitted bundary function and a size parameter. The size parameter is used to calculate the initial universe of points, which are then passed to the function

        Args:
            eval_boundary (function): Jitted function representing the boundary. It should return True is the boundary condition is satisfied, False otherwise.
            size (int): lenght of the side of the square that constitue the universe of admissible points

        Returns:
            list: list of all the points that satisfy the boundary, as tuples
        """
        lattice = np.arange(-size, size)
        points = []
        for i in range(len(lattice)):
            for j in range(len(lattice)):
                x = lattice[i]
                y = lattice[j]
                if eval_boundary(x, y):
                    points.append((x, y))
        return points

    @staticmethod
    def _check_if_bounday_is_touched(points, n) -> bool:
        """For generating our list of admissible points (the points that live inside the boundary) we adopt a naive approach, considering a given n*n square and evaluating which of the points satisfy the boundary. This function is used to determined wether the square used is big enough: if at least one of our admissible points touches the edges of our boundary, we need to increase the n*n square.

        Args:
            points (list): list of tuples containing the admissible points
            n (int): size of the side of the square

        Returns:
            bool: Returns True if any point touches the boundary. False otherwise.
        """
        if points is None:
            return True
        coordinates = [item for t in points for item in t]
        max_el = max(coordinates)
        min_el = min(coordinates)
        if (abs(max_el) >= n) or (abs(min_el) >= n):
            return True
        return False

    @staticmethod
    @lru_cache(maxsize=None)
    def _get_adiacent(x, y) -> set:
        """Returns the set of points that are adiacent to a given (x,y) point.
        """
        return set([(x-1, y), (x, y-1), (x+1, y), (x, y+1)])

    @staticmethod
    def take_step() -> int:
        rnd = random.uniform(0, 1)
        if rnd < 0.5:
            return -1
        return 1

    """ Class Methods """

    def show_transition_points(self) -> None:
        """Used to generate a visual of the admissible points.
        """
        plt.scatter(*zip(*self.points))
        plt.show()

    def _create_admissible_points(self) -> list:
        """Here we generate the list of all points that meet the boundary condition.

        Returns:
            list: List of tuples containing all the admissible points.
        """
        n = 1
        points = None
        while self._check_if_bounday_is_touched(points, n):
            n = n*10
            points = self._get_points(eval_boundary=self.eval_bound, size=n)
        return points

    def _get_markov_matrices(self) -> None:
        """Create the relevant markov matrices for the problem at hand.
        """
        points_i = [f'{t[0]},{t[1]}' for t in self.points]
        points_s = set(self.points)
        transition_matrix = pd.DataFrame(
            index=points_i, columns=points_i).fillna(0)
        for point in self.points:
            x, y = point
            admissible_adiacents = list(
                self._get_adiacent(x, y).intersection(points_s))
            for adiacent in admissible_adiacents:
                xa, ya = str(adiacent[0]), str(adiacent[1])
                transition_matrix.loc[f'{x},{y}', f'{xa},{ya}'] = 0.25

        identity = np.identity(len(points_i))
        n = pd.DataFrame(np.linalg.inv(
            (identity-transition_matrix.values)), index=points_i, columns=points_i)
        solutions = n.dot(np.ones((len(points_i), 1))).rename(
            columns={0: 'avg_steps'})

        self.transition_matrix = transition_matrix
        self.solutions = solutions

    def get_markov_solution(self) -> None:
        """Initializes markov matrices and retrieves and print the desired solution.
        """
        self._get_markov_matrices()
        avg_steps = self.solutions.loc[f'{self.x0},{self.y0}', 'avg_steps']
        print(
            f'Bill has to make {avg_steps} steps on average to reach the food. Poor Bill!')

    def do_run(self) -> int:
        """Used to simulate a single run to food, based on the starting position of the ant.

        Returns:
            int: The number of steps that the run took.
        """
        x = self.x0
        y = self.y0
        steps = 0
        while self.eval_bound(x, y):
            rnd = random.uniform(0, 1)
            if rnd < 0.5:
                x += self.take_step()
            else:
                y += self.take_step()
            steps += 1
        return steps

    def calculate_avg_steps(self, n_runs=1000) -> float:
        """Calculate the average number of steps the ant has made over n_runs

        Args:
            n_runs (int, optional): Number of runs to food, oer which we calculate the average. Defaults to 1000.

        Returns:
            float: The average number of steps.
        """
        steps = [self.do_run() for _ in range(n_runs)]
        return sum(steps)/len(steps)

    def do_monte_carlo_avg_steps(self, n_samples=100, n_runs=1000) -> pd.DataFrame():
        """We do a monte carlo simulation for the mean, that is, the average number of steps the ant has to take in order to reach the food. More precisely, we simulate n_runs an n_samples amount of times (basically, we create n_sample observations for the mean).

        Returns:
            pd.DataFrame: The dataframe containing the average steps for each sample.
        """
        if self.workers is not None:
            with multiprocessing.Pool(processes=self.workers) as pool:
                avg_steps = pool.map(self.calculate_avg_steps, [
                                     n_runs for _ in range(n_samples)])
        else:
            avg_steps = [self.calculate_avg_steps(
                n_runs=n_runs) for _ in range(n_samples)]
        data = pd.DataFrame(data=avg_steps, columns=['average_steps'])
        print(
            f'Bill has to make {data["average_steps"].mean()} steps on average to reach the food. Poor Bill!')
        return data

    @lru_cache(maxsize=None)
    def _get_probability_recusive(self, t, p):
        """Given a time and a point (x,y), this function calculates recursively the probability of being in (x,y) at time t. This probability is calculated as: p(t,(x,y)) = 0.25(sum(p(t-1,adiacent))) over all adiacent points.

        Args:
            t (int): time
            p (tuple): point

        Returns:
            float: probability
        """
        # base cases
        if (t == 0) and (p == (0, 0)):
            return 1
        if t == 0:
            return 0
        # get adiacent points and calculate recursively
        adiacents = self._get_adiacent(p[0], p[1])
        admissible_adiacents = list(set(self.points).intersection(adiacents))
        return sum([0.25*self._get_probability_recusive(t-1, adiacent) for adiacent in admissible_adiacents])

    def _get_first_non_transition_points(self):
        """This function calculates the immediately-outer boundary to the transition points. We could call this the set of 'arrival points'.

        Returns:
            list: all arrival points
        """
        all_points = set()
        for point in self.points:
            all_points = all_points.union(
                self._get_adiacent(point[0], point[1]))
        return list(all_points - set(self.points))

    def calculate_avg_steps_recursive(self, tol=0.0001):
        # stopping condition and arrival points
        first_non_transition_points = self._get_first_non_transition_points()
        t = 0
        cum_p = 0
        data_dict = {}

        while (1-cum_p) > tol:
            p_t = []
            for point in first_non_transition_points:
                p_t.append(self._get_probability_recusive(t, point))
            t_sum = sum(p_t)
            data_dict[t] = t_sum
            t += 1
            cum_p += t_sum

        data = pd.DataFrame.from_dict(
            data_dict, orient='index', columns=['p_t'])
        data.index.name = 'steps'
        data = data.reset_index(drop=False)
        print(
            f'Bill has to make {(data["p_t"] * data["steps"]).sum()} steps on average to reach the food. Poor Bill!')
        return (data['p_t'] * data['steps']).sum(), data
