"""This module implements the CMA-ES algorithm for hyperparameter tuning in Keras Tuner."""

from __future__ import annotations

# Import necessary modules
from typing import Callable
from typing import Optional

import numpy as np

from keras_tuner.engine import oracle as oracle_module
from keras_tuner.engine import trial as trial_module
from keras_tuner.engine import tuner as tuner_module
from keras_tuner.src.api_export import keras_tuner_export

from keras_tuner_extensionpack.tools.converters import array2hp
from keras_tuner_extensionpack.tools.converters import hp2array
from keras_tuner_extensionpack.tools.type_definitions import HP
from keras_tuner_extensionpack.tools.type_definitions import TrialDict


# Define CMAESAlgorithmOracle class
class CMAESAlgorithmOracle(oracle_module.Oracle):
    """Conjunctive Normal Form Evolution Strategy (CMA-ES) Oracle.

    Args:
        objective (Optional[Callable]): The objective function to optimize.
            It can be a callable function or a list of callable functions.
            If a list is provided, the objectives will be summed for
            minimization or subtracted for maximization.
        trials_size (int): The maximum number of trials (model configurations)
            to test. The oracle may interrupt the search before reaching this
            limit if the search space is exhausted.
        population_size (int): The size of the population used by the CMA-ES
            algorithm.
        sigma_init (float): The initial value of the step size.
        epsilon (float): A small value used for numerical stability.
        threshold (float): The threshold value for convergence.
        seed: Optional integer, the random seed.
        hyperparameters: Optional `HyperParameters` instance. Can be used to
            override (or register in advance) hyperparameters in the search
            space.
        tune_new_entries: Boolean, whether hyperparameter entries that are
            requested by the hypermodel but that were not specified in
            `hyperparameters` should be added to the search space, or not. If
            not, then the default value for these parameters will be used.
            Defaults to True.
        allow_new_entries: Boolean, whether the hypermodel is allowed to
            request hyperparameter entries not listed in `hyperparameters`.
            Defaults to True.
        max_retries_per_trial: Integer. Defaults to 0. The maximum number of
            times to retry a `Trial` if the trial crashed or the results are
            invalid.
        max_consecutive_failed_trials: Integer. Defaults to 3. The maximum
            number of consecutive failed `Trial`s. When this number is reached,
            the search will be stopped. A `Trial` is marked as failed when none
            of the retries succeeded.

    Attributes:
        trials_size (int): The maximum number of trials (model configurations)
            to test.
        population_size (int): The size of the population used by the CMA-ES
            algorithm.
        sigma (float): The current value of the step size.
        sigma_init (float): The initial value of the step size.
        epsilon (float): A small value used for numerical stability.
        threshold (float): The threshold value for convergence.
        population (np.ndarray): An array representing the population of
            hyperparameter configurations.
        solutions (np.ndarray): An array representing the solutions generated
            by the CMA-ES algorithm.
        fitness (np.ndarray): An array representing the fitness values of the
            population.
        best_fitness (float): The best fitness value found so far.
        init_seed (Optional[int]): The initial random seed.
        finial_seed (Optional[int]): The final random seed.
        mean (np.ndarray): The mean vector of the population.
        cov (np.ndarray): The covariance matrix of the population.
        dim (int): The dimensionality of the search space.
        p_sigma (np.ndarray): The evolution path for the step size.
        p_c (np.ndarray): The evolution path for the covariance matrix.
        mu (float): The number of parents selected for recombination.
        weights (float): The weights used for recombination.
        mu_eff (float): The effective selection mass.
        cc (float): The time constant for cumulation for the covariance matrix.
        cs (float): The time constant for cumulation for the step size.
        c1 (float): The learning rate for the rank-one update of the covariance
            matrix.
        cmu (float): The learning rate for the rank-mu update of the covariance
            matrix.
        damps (float): The damping factor for step size adaptation.
        old_solution (np.ndarray): The previous best solution.
        method (CMAESMethod): The method used for CMA-ES.

    """

    def __init__(
        self,
        objective: Optional[Callable] = None,
        trials_size: int = 100,
        population_size: int = 20,
        sigma_init: float = 0.5,
        epsilon: float = 1e-8,
        threshold: float = 1.4,
        hyperparameters: Optional[HP] = None,
        *,
        allow_new_entries: bool = True,
        tune_new_entries: bool = True,
        seed: Optional[int] = None,
        max_retries_per_trial: int = 0,
        max_consecutive_failed_trials: int = 3,
    ) -> None:
        # Initialize attributes
        self.trials_size = trials_size
        self.population_size = population_size
        self.sigma = self.sigma_init = sigma_init
        self.epsilon = epsilon
        self.threshold = threshold
        max_trials = self.trials_size * self.population_size

        # Call super constructor
        super().__init__(
            objective,
            max_trials,
            hyperparameters,
            allow_new_entries,
            tune_new_entries,
            seed,
            max_retries_per_trial,
            max_consecutive_failed_trials,
        )

        self.population: np.ndarray = np.array([{}])
        self.solutions: np.ndarray = np.array([])
        self.fitness = np.empty(self.population_size)
        self.best_fitness = np.inf

        self.init_seed = self.finial_seed = seed
        self.mean: np.ndarray = np.array([])
        self.cov: np.ndarray = np.array([])
        self.dim: int = 1
        self.p_sigma: np.ndarray = np.array([])
        self.p_c: np.ndarray = np.array([])
        self.mu = 0.0
        self.weights = 0.0
        self.mu_eff = 0.0
        self.cc = 0.0
        self.cs = 0.0
        self.c1 = 0.0
        self.cmu = 0.0
        self.damps = 0.0
        self.old_solution = np.array([])

    def initialize_parameters(self) -> None:
        """Initialize the parameters of the CMA-ES algorithm."""
        self.dim = len(self.hyperparameters.space)
        self.mean = hp2array(self.hyperparameters, self._random_values())
        self.cov = np.eye(self.dim)
        self.p_sigma = np.zeros(self.dim)
        self.p_c = np.zeros(self.dim)
        self.mu = self.population_size // 2
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights /= np.sum(self.weights)
        self.mu_eff = 1 / np.sum(self.weights**2)
        self.cc = (4 + self.mu_eff / self.dim) / (
            self.dim + 4 + 2 * self.mu_eff / self.dim
        )
        self.cs = (self.mu_eff + 2) / (self.dim + self.mu_eff + 5)
        self.c1 = 2 / ((self.dim + 1.3) ** 2 + self.mu_eff)
        self.cmu = min(
            1 - self.c1,
            2
            * (self.mu_eff - 2 + 1 / self.mu_eff)
            / ((self.dim + 2) ** 2 + self.mu_eff),
        )
        self.damps = (
            1 + 2 * max(0, np.sqrt((self.mu_eff - 1) / (self.dim + 1)) - 1) + self.cs
        )

    def populate_space(self, trial_id: str) -> TrialDict:
        """Populates the search space for a given trial.

        Args:
            trial_id (str): The ID of the trial.

        Returns:
            TrialDict: A dictionary containing the status of the trial and the values
                of the search space.
        """
        if int(trial_id) == 0:
            self.init_seed = self.seed
            return {"status": trial_module.TrialStatus.RUNNING, "values": {}}

        self.seed += 1

        if int(trial_id) == 1:
            self.initialize_parameters()
            self.old_solution = self.mean
            return {
                "status": trial_module.TrialStatus.RUNNING,
                "values": array2hp(self.hyperparameters, self.old_solution),
            }

        j = int(trial_id) % self.population_size
        if j - 2 == 0:
            self._generate_and_normalize_solutions()
            self._update_solutions_and_population()

        if j - 2 == self.population_size - 1:
            self._inner_loop(trial_id)

        self._update_fitness_and_best_values(j)

        return {
            "status": trial_module.TrialStatus.RUNNING,
            "values": self.population[j],
        }

    def _generate_and_normalize_solutions(self) -> None:
        """Generate and normalize the solutions."""
        self.solutions = np.random.default_rng(self.seed).multivariate_normal(
            self.mean,
            self.sigma**2 * self.cov,
            self.population_size,
        )
        self.solutions = (self.solutions - self.solutions.min()) / (
            self.solutions.max() - self.solutions.min()
        )

    def _update_solutions_and_population(self) -> None:
        """Update the solutions and population."""
        self.old_solution = self.solutions[-1]
        self.solutions = np.roll(self.solutions, 1)
        self.solutions[0] = self.old_solution
        self.population = np.array(
            [array2hp(self.hyperparameters, solution) for solution in self.solutions]
        )

    def _update_fitness_and_best_values(self, j: int) -> None:
        """Update the fitness and best values."""
        self.fitness[j] = self.trials[self.start_order[-1]].score

    def _inner_loop(self, trial_id: str) -> None:
        """Update the parameters of the CMA-ES algorithm.

        Args:
            trial_id (str): The ID of the trial.
        """
        indices = self.fitness.argsort()
        mean_old = self.mean
        self.mean = np.dot(self.weights, self.solutions[indices[: self.mu]])
        self.p_sigma = (1 - self.cs) * self.p_sigma + np.sqrt(
            self.cs * (2 - self.cs) * self.mu_eff,
        ) * np.dot(
            np.linalg.inv(np.linalg.cholesky(self.cov)),
            (self.mean - mean_old) / self.sigma,
        )
        h_sigma = (
            np.linalg.norm(self.p_sigma)
            / np.sqrt(1 - (1 - self.cs) ** (2 * (int(trial_id) + 1)))
            / np.sqrt(self.dim)
            < self.threshold
        )
        self.p_c = (1 - self.cc) * self.p_c + h_sigma * np.sqrt(
            self.cc * (2 - self.cc) * self.mu_eff,
        ) * (self.mean - mean_old) / self.sigma
        artmp = (1 / self.sigma) * (self.solutions[indices[: self.mu]] - mean_old)
        self.cov = (
            (1 - self.c1 - self.cmu) * self.cov
            + self.c1
            * (
                np.outer(self.p_c, self.p_c)
                + (1 - h_sigma) * self.cc * (2 - self.cc) * self.cov
            )
            + self.cmu * np.dot(artmp.T, np.dot(np.diag(self.weights), artmp))
        )
        self.sigma *= np.exp(
            (self.cs / self.damps)
            * (np.linalg.norm(self.p_sigma) / np.sqrt(self.dim) - 1),
        )
        self.sigma = max(self.sigma, self.epsilon)
        self.sigma *= np.exp(
            (self.cs / self.damps)
            * (np.linalg.norm(self.p_sigma) / np.sqrt(self.dim) - 1),
        )

    def get_state(self) -> dict:
        """Get the state of the oracle.

        Returns:
            dict: A dictionary containing the state of the oracle.
        """
        state = super().get_state()
        # Add additional state information specific to CMA-ES algorithm if needed
        state.update(
            {
                "trials_size": self.trials_size,
                "population_size": self.population_size,
                "epsilon": self.epsilon,
                "sigma_init": self.sigma_init,
                "threshold": self.threshold,
                "fitness": self.fitness.tolist(),
                "init_seed": self.init_seed,
                "finial_seed": self.seed,
            },
        )
        return state

    def set_state(self, state) -> None:
        """Set the state of the oracle."""
        super().set_state(state)
        self.trials_size = state["trials_size"]
        self.population_size = state["population_size"]
        self.epsilon = state["epsilon"]
        self.sigma_init = state["sigma_init"]
        self.threshold = state["threshold"]
        self.fitness = state["fitness"]
        self.init_seed = state["init_seed"]
        self.finial_seed = state["finial_seed"]


@keras_tuner_export(
    [
        "keras_tuner.CMAESAlgorithm",
        "keras_tuner.tuners.CMAESAlgorithm",
    ],
)
class CMAESAlgorithm(tuner_module.Tuner):
    """CMAESAlgorithm.

    CMAESAlgorithm is a subclass of `tuner_module.Tuner` that implements the
    Covariance Matrix Adaptation Evolution Strategy (CMA-ES) algorithm for
    hyperparameter tuning.

    Args:
        hypermodel: Instance of `HyperModel` class (or callable that takes
            hyperparameters and returns a `Model` instance). It is optional
            when `Tuner.run_trial()` is overridden and does not use
            `self.hypermodel`.
        objective: A string, `keras_tuner.Objective` instance, or a list of
            `keras_tuner.Objective`s and strings. If a string, the direction of
            the optimization (min or max) will be inferred. If a list of
            `keras_tuner.Objective`, we will minimize the sum of all the
            objectives to minimize subtracting the sum of all the objectives to
            maximize. The `objective` argument is optional when
            `Tuner.run_trial()` or `HyperModel.fit()` returns a single float as
            the objective to minimize.
        trials_size: Integer, the number of trials to be generated in each
            iteration of the algorithm.
        population_size: Integer, the number of individuals in the population.
        sigma_init: Float, the initial standard deviation for the search
            distribution.
        epsilon: Float, small constant used to avoid division by zero.
        threshold: Float, the threshold for convergence.
        seed: Optional integer, the random seed.
        hyperparameters: Optional `HyperParameters` instance. Can be used to
            override (or register in advance) hyperparameters in the search
            space.
        tune_new_entries: Boolean, whether hyperparameter entries that are
            requested by the hypermodel but that were not specified in
            `hyperparameters` should be added to the search space, or not. If
            not, then the default value for these parameters will be used.
            Defaults to True.
        allow_new_entries: Boolean, whether the hypermodel is allowed to
            request hyperparameter entries not listed in `hyperparameters`.
            Defaults to True.
        max_retries_per_trial: Integer. Defaults to 0. The maximum number of
            times to retry a `Trial` if the trial crashed or the results are
            invalid.
        max_consecutive_failed_trials: Integer. Defaults to 3. The maximum
            number of consecutive failed `Trial`s. When this number is reached,
            the search will be stopped. A `Trial` is marked as failed when none
            of the retries succeeded.
        **kwargs: Keyword arguments relevant to all `Tuner` subclasses. Please
            see the docstring for `Tuner`.

    Raises:
        ValueError: If `hypermodel` is not provided.
    """

    def __init__(
        self,
        hypermodel=None,
        objective=None,
        trials_size=None,
        population_size=None,
        sigma_init=0.5,
        epsilon=1e-8,
        threshold=1.4,
        hyperparameters=None,
        allow_new_entries=True,
        tune_new_entries=True,
        seed=None,
        max_retries_per_trial=0,
        max_consecutive_failed_trials=3,
        **kwargs,
    ) -> None:
        oracle = CMAESAlgorithmOracle(
            objective=objective,
            trials_size=trials_size,
            population_size=population_size,
            sigma_init=sigma_init,
            epsilon=epsilon,
            threshold=threshold,
            hyperparameters=hyperparameters,
            allow_new_entries=allow_new_entries,
            tune_new_entries=tune_new_entries,
            seed=seed,
            max_retries_per_trial=max_retries_per_trial,
            max_consecutive_failed_trials=max_consecutive_failed_trials,
        )
        super().__init__(oracle=oracle, hypermodel=hypermodel, **kwargs)
