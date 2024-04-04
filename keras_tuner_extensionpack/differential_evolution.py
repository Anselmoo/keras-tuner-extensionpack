"""This module implements the Differential Evolution algorithm for hyperparameter optimization."""

from __future__ import annotations

from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Callable
from typing import Optional
from typing import Tuple
import numpy as np

from keras_tuner.engine import oracle as oracle_module
from keras_tuner.engine import trial as trial_module
from keras_tuner.engine import tuner as tuner_module
from keras_tuner.src.api_export import keras_tuner_export
from keras_tuner.src.engine import hyperparameters as hp_module

from keras_tuner_extensionpack.tools import exception_checks
from keras_tuner_extensionpack.tools.type_definitions import HP
from keras_tuner_extensionpack.tools.type_definitions import Hyperparams
from keras_tuner_extensionpack.tools.type_definitions import TrialDict


class DifferentialEvolutionMethod:
    """Implements the Differential Evolution method for optimization.

    Attributes:
        None

    Methods:
        select: Selects a portion of the population based on fitness.
        model: Estimates the mean and standard deviation of the population.
        sample: Generates new individuals by sampling from the model.
    """

    def select(
        self, fitness: np.ndarray, population_size: int, population: np.ndarray
    ) -> np.ndarray:
        """Selects a portion of the population with the best fitness.

        Args:
            fitness (np.ndarray): An array of fitness values for each individual in the population.
            population_size (int): The size of the population.
            population (np.ndarray): An array representing the current population.

        Returns:
            np.ndarray: An array representing the selected portion of the population.
        """
        idx = np.argsort(fitness)[: population_size // 2]
        return population[idx]

    def model(
        self, population: np.ndarray, hyperparameters: HP
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Estimates the mean and standard deviation of the population.

        Args:
            population (np.ndarray): An array representing the current population.
            hyperparameters (HP): The hyperparameters used for optimization.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the mean and standard deviation of the population.
        """
        _population = np.empty(
            (len(population), len(hyperparameters.space)),
            dtype=float,
        )
        for i, d in enumerate(population):
            hps = hp_module.HyperParameters()
            for j, hp in enumerate(hyperparameters.space):
                hps.merge([hp])
                if hps.is_active(hp):
                    _population[i, j] = hp.value_to_prob(d[hp.name])

        mean = _population.mean(axis=0)
        std = _population.std(axis=0)
        return mean, std

    def sample(
        self,
        seed: int,
        population_size: int,
        hyperparameters: HP,
        mean: np.ndarray,
        std: np.ndarray,
    ) -> np.ndarray:
        """Generates new individuals by sampling from the model.

        Args:
            seed (int): The seed value for random number generation.
            population_size (int): The size of the population.
            hyperparameters (HP): The hyperparameters used for optimization.
            mean (np.ndarray): The mean of the population.
            std (np.ndarray): The standard deviation of the population.

        Returns:
            np.ndarray: An array representing the newly generated population.
        """
        population = []
        hps = hp_module.HyperParameters()

        random_normal_values = np.random.default_rng(seed).normal(
            mean,
            std,
            (population_size, len(hyperparameters.space)),
        )
        for i in range(population_size):
            for _value, hp in zip(random_normal_values[i], hyperparameters.space):
                hps.merge([hp])
                if hps.is_active(hp) and not isinstance(hp, hp_module.Fixed):
                    _value = hp.prob_to_value(np.clip(_value, 0, 1))
                    hps.values[hp.name] = _value

            population.append(hps.values.copy())
            hps.values.clear()
        return np.array(population)


@keras_tuner_export("keras_tuner.oracles.DifferentialEvolutionOracle")
class DifferentialEvolutionOracle(oracle_module.Oracle):
    """An optimizer oracle that uses the Differential Evolution algorithm.

    Args:
        objective: A string, `keras_tuner.Objective` instance, or a list of
            `keras_tuner.Objective`s and strings. If a string, the direction of
            the optimization (min or max) will be inferred. If a list of
            `keras_tuner.Objective`, we will minimize the sum of all the
            objectives to minimize subtracting the sum of all the objectives to
            maximize. The `objective` argument is optional when
            `Tuner.run_trial()` or `HyperModel.fit()` returns a single float as
            the objective to minimize.
        trials_size (int): The number of trials to run.
        population_size (int): The size of the population.
        elitism_rate (float): The rate of elites in the population.
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
        trials_size (int): The number of trials to run.
        population_size (int): The size of the population.
        elitism_rate (float): The rate of elites in the population.
        population (np.ndarray): The current population of candidate solutions.
        fitness (np.ndarray): The fitness values of the candidate solutions.
        best_fitness (float): The best fitness value found so far.
        best_values (Hyperparams): The best set of hyperparameters found so far.
        init_seed (Optional[int]): The initial seed for random number generation.
        finial_seed (Optional[int]): The final seed for random number generation.
        method (DifferentialEvolutionMethod): The method used for differential evolution.

    Methods:
        populate_space(trial_id: str) -> TrialDict:
            Fill the hyperparameter space with values for the given trial.
        _inner_loop() -> None:
            Perform the inner loop of the differential evolution algorithm.
        _update_fitness_and_best_values(trial_id: str) -> None:
            Update the fitness and best values based on the trial ID.
        get_state() -> Dict[str, Any]:
            Get the state of the DifferentialEvolutionOracle.
        set_state(state: Dict[str, Any]) -> None:
            Set the state of the DifferentialEvolutionOracle.
    """

    def __init__(
        self,
        objective: Optional[Callable] = None,
        trials_size: int = 100,
        population_size: int = 20,
        elitism_rate: float = 0.1,
        hyperparameters: Optional[HP] = None,
        *,
        allow_new_entries: bool = True,
        tune_new_entries: bool = True,
        seed: Optional[int] = None,
        max_retries_per_trial: int = 0,
        max_consecutive_failed_trials: int = 3,
    ):
        """Initialize the DifferentialEvolutionOracle.

        Args:
            objective (Optional[Callable]): The objective function to optimize.
            trials_size (int): The number of trials to run.
            population_size (int): The size of the population.
            elitism_rate (float): The rate of elites in the population.
            hyperparameters (Optional[HP]): The hyperparameters to tune.
            allow_new_entries (bool): Whether to allow new hyperparameter entries.
            tune_new_entries (bool): Whether to tune new hyperparameter entries.
            seed (Optional[int]): The seed for random number generation.
            max_retries_per_trial (int): The maximum number of retries per trial.
            max_consecutive_failed_trials (int): The maximum number of consecutive failed trials.
        """
        self.trials_size = trials_size
        self.population_size = population_size
        max_trials = self.trials_size * self.population_size + 1
        self.elitism_rate = elitism_rate
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
        self.fitness = np.empty(self.population_size)
        self.best_fitness = np.inf
        self.best_values: Hyperparams = {}
        self.init_seed = seed
        self.finial_seed = seed
        self.method = DifferentialEvolutionMethod()

    def populate_space(self, trial_id: str) -> TrialDict:
        """Fill the hyperparameter space with values for the given trial.

        Args:
            trial_id (str): The ID of the trial.

        Returns:
            TrialDict: The trial status and the current trial's hyperparameters.
        """
        if int(trial_id) == 0:
            self.init_seed = self.seed
            return {"status": trial_module.TrialStatus.RUNNING, "values": {}}
        # Change the seed for each trial.
        self.seed += 1
        if int(trial_id) == 1:
            # Initialize the hyperparameters set with random values
            # according to the population size.
            self.population = np.array(
                [self._random_values() for _ in range(self.population_size)],
            )

        j = int(trial_id) % self.population_size
        if int(trial_id) >= 1 and j == 0:
            self._inner_loop()
        # Store the current trial's hyperparameters.
        current_values = self.population[j]
        # Update the fitness of the previous trial and the best values.
        self._update_fitness_and_best_values(trial_id)

        # Update the hyperparameters set with new values.
        if self.best_values is None:
            return {"status": trial_module.TrialStatus.STOPPED, "values": None}

        # Return the current trial's hyperparameters.
        return {"status": trial_module.TrialStatus.RUNNING, "values": current_values}

    def _inner_loop(self) -> None:
        """Perform the inner loop of the differential evolution algorithm."""
        elites = self.population[
            np.argsort(self.fitness)[: int(self.elitism_rate * self.population_size)],
        ]
        self.population = self.method.select(
            self.fitness,
            self.population_size,
            self.population,
        )
        mean, std = self.method.model(self.population, self.hyperparameters)
        self.population = self.method.sample(
            self.seed,
            self.population_size,
            self.hyperparameters,
            mean,
            std,
        )
        self.population[: len(elites)] = elites

    def _update_fitness_and_best_values(self, trial_id: str) -> None:
        """Update the fitness and best values based on the trial ID.

        Args:
            trial_id (str): The ID of the trial.
        """
        # Update the fitness of the previous trial.
        if int(trial_id) > 0:
            self.fitness[(int(trial_id) - 1) % self.population_size] = self.trials[
                self.start_order[-1]
            ].score

        # Update the best fitness and values.
        if self.fitness[(int(trial_id) - 1) % self.population_size] < self.best_fitness:
            self.best_fitness = self.fitness[(int(trial_id) - 1) % self.population_size]
            self.best_values = self.population[
                (int(trial_id) - 1) % self.population_size
            ]

    def get_state(self) -> Dict[str, Any]:
        """Get the state of the DifferentialEvolutionOracle.

        Returns:
            Dict[str, Any]: The state of the oracle.
        """
        state = super().get_state()
        state.update(
            {
                "trials_size": self.trials_size,
                "population_size": self.population_size,
                "elitism_rate": self.elitism_rate,
                "fitness": self.fitness.tolist(),
                "best_fitness": float(self.best_fitness),
                "best_values": self.best_values,
                "init_seed": self.init_seed,
                "finial_seed": self.seed,
            },
        )
        return state

    def set_state(self, state: Dict[str, Any]) -> None:
        """Set the state of the DifferentialEvolutionOracle.

        Args:
            state (Dict[str, Any]): The state of the oracle.
        """
        super().set_state(state)
        self.trials_size = state["trials_size"]
        self.population_size = state["population_size"]
        self.elitism_rate = state["elitism_rate"]
        self.fitness = state["fitness"]
        self.best_fitness = state["best_fitness"]
        self.best_values = state["best_values"]
        self.init_seed = state["init_seed"]
        self.finial_seed = state["finial_seed"]


@keras_tuner_export(
    [
        "keras_tuner.DifferentialEvolution",
        "keras_tuner.tuners.DifferentialEvolution",
    ],
)
class DifferentialEvolution(tuner_module.Tuner):
    """Differential Evolution algorithm for hyperparameter optimization.

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
        trials_size (Optional[int]): The number of trials to be generated in each generation of the Differential Evolution algorithm.
        population_size (Optional[int]): The number of individuals in each generation of the Differential Evolution algorithm.
        elitism_rate (float): The proportion of top individuals to be preserved in each generation as elite individuals.
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
    """

    def __init__(
        self,
        hypermodel=None,
        objective=None,
        trials_size=None,
        population_size=None,
        elitism_rate=0.1,
        hyperparameters=None,
        allow_new_entries=True,
        tune_new_entries=True,
        seed=None,
        max_retries_per_trial=0,
        max_consecutive_failed_trials=3,
        **kwargs,
    ):
        oracle = DifferentialEvolutionOracle(
            objective=objective,
            trials_size=trials_size,
            population_size=population_size,
            elitism_rate=elitism_rate,
            hyperparameters=hyperparameters,
            allow_new_entries=allow_new_entries,
            tune_new_entries=tune_new_entries,
            seed=seed,
            max_retries_per_trial=max_retries_per_trial,
            max_consecutive_failed_trials=max_consecutive_failed_trials,
        )
        super().__init__(oracle=oracle, hypermodel=hypermodel, **kwargs)
