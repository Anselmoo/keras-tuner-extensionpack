"""This module implements the Harmony Search algorithm for hyperparameter optimization."""

from __future__ import annotations

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


@keras_tuner_export("keras_tuner.oracles.HarmonySearchOracle")
class HarmonySearchOracle(oracle_module.Oracle):
    """Optimizer oracle using the Harmony Search algorithm.

    Args:
        objective: A string, `keras_tuner.Objective` instance, or a list of
            `keras_tuner.Objective`s and strings. If a string, the direction of
            the optimization (min or max) will be inferred. If a list of
            `keras_tuner.Objective`, we will minimize the sum of all the
            objectives to minimize subtracting the sum of all the objectives to
            maximize. The `objective` argument is optional when
            `Tuner.run_trial()` or `HyperModel.fit()` returns a single float as
            the objective to minimize.
        trials_size (int, optional): The number of trials to run.
        population_size (int, optional): The size of the population.
        harmony_memory_accepting_rate (float, optional): The rate at which the harmony
            memory is accepted. Should be a value between 0 and 1.
        pitch_adjusting_rate (float, optional): The rate at which the pitch is adjusted.
            Should be a value between 0 and 1.
        bandwidth (float, optional): The bandwidth for adjusting the pitch.
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
        harmony_memory_accepting_rate (float): The rate at which the harmony memory is
            accepted.
        pitch_adjusting_rate (float): The rate at which the pitch is adjusted.
        bandwidth (float): The bandwidth for adjusting the pitch.
        population (np.ndarray): The population of harmonies.
        fitness (np.ndarray): The fitness values of the harmonies.
        best_fitness (float): The best fitness value found so far.
        best_values (Hyperparams): The best set of hyperparameters found so far.
        init_seed (int): The initial seed for random number generation.
        finial_seed (int): The final seed for random number generation.

    Methods:
        populate_space(trial_id: str) -> dict:
            Fill the hyperparameter space with values for a given trial.

        _inner_loop() -> tuple[Hyperparams, Hyperparams]:
            Perform the inner loop of the Harmony Search algorithm.

        _update_fitness_and_best_values(previous_values: Hyperparams) -> None:
            Update the fitness values and best hyperparameter values.

        get_state() -> dict:
            Get the state of the HarmonySearchOracle.

        set_state(state: dict) -> None:
            Set the state of the HarmonySearchOracle.
    """

    def __init__(
        self,
        objective: Optional[Callable] = None,
        trials_size: int = 100,
        population_size: int = 20,
        harmony_memory_accepting_rate: float = 0.95,
        pitch_adjusting_rate: float = 0.7,
        bandwidth: float = 0.01,
        hyperparameters: Optional[HP] = None,
        allow_new_entries: bool = True,
        tune_new_entries: bool = True,
        seed: Optional[int] = None,
        max_retries_per_trial: int = 0,
        max_consecutive_failed_trials: int = 3,
    ) -> None:
        """Initialize the HarmonySearchOracle.

        Args:
            objective (callable, optional): The objective function to optimize.
            trials_size (int, optional): The number of trials to run.
            population_size (int, optional): The size of the population.
            harmony_memory_accepting_rate (float, optional): The rate at which the
                harmony memory is accepted.
            pitch_adjusting_rate (float, optional): The rate at which the pitch is
                adjusted.
            bandwidth (float, optional): The bandwidth for adjusting the pitch.
            hyperparameters (HyperParameters, optional): The hyperparameters to tune.
            allow_new_entries (bool, optional): Whether to allow new entries in the
                hyperparameters.
            tune_new_entries (bool, optional): Whether to tune new entries in the
                hyperparameters.
            seed (int, optional): The seed for random number generation.
            max_retries_per_trial (int, optional): The maximum number of retries per
                trial.
            max_consecutive_failed_trials (int, optional): The maximum number of
                consecutive failed trials.
        """
        self.trials_size = trials_size
        self.population_size = population_size

        self.harmony_memory_accepting_rate = harmony_memory_accepting_rate
        self.pitch_adjusting_rate = pitch_adjusting_rate
        self.bandwidth = bandwidth

        exception_checks.not_in_range_check(
            "harmony_memory_accepting_rate",
            harmony_memory_accepting_rate,
        )
        exception_checks.not_in_range_check(
            "pitch_adjusting_rate",
            pitch_adjusting_rate,
        )
        exception_checks.not_in_range_check(
            "bandwidth",
            bandwidth,
            lower_bound_include=True,
        )

        max_trials = self.trials_size * self.population_size

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

    def populate_space(self, trial_id: str) -> TrialDict:
        """Fill the hyperparameter space with values.

        Args:
            trial_id (str): The ID of the trial.

        Returns:
            dict: A dictionary containing the status and values for the trial.
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
            return {
                "status": trial_module.TrialStatus.RUNNING,
                "values": self.population[0],
            }

        if int(trial_id) < (self.population_size + 1):
            self.fitness[int(trial_id) - 2] = self.trials[self.start_order[-1]].score
            return {
                "status": trial_module.TrialStatus.RUNNING,
                "values": self.population[int(trial_id) - 1],
            }

        if int(trial_id) == (self.population_size + 1):
            # define the current best and score
            self.fitness[int(trial_id) - 2] = self.trials[self.start_order[-1]].score
            best_idx = np.argmin(self.fitness)
            self.best_values = self.population[best_idx]
            self.best_fitness = self.fitness[best_idx]
            return {
                "status": trial_module.TrialStatus.RUNNING,
                "values": self.best_values,
            }

        current_values, previous_values = self._inner_loop()
        self._update_fitness_and_best_values(previous_values)

        if self.best_values is None:
            return {"status": trial_module.TrialStatus.STOPPED, "values": None}
        return {
            "status": trial_module.TrialStatus.RUNNING,
            "values": current_values,
        }

    def _inner_loop(self) -> Tuple[Hyperparams, Hyperparams]:
        """Perform the inner loop of the Harmony Search algorithm.

        This method executes the inner loop of the Harmony Search algorithm,
        which is responsible for generating new hyperparameter values based on
        the harmony memory and adjusting the values using pitch adjusting rate.

        Returns:
            Tuple[Hyperparams, Hyperparams]: The current and previous hyperparameter values.
        """
        previous_values = self.best_values.copy()
        harmony_memory = self.population.copy()
        hps = hp_module.HyperParameters()

        for i, hp in enumerate(self.hyperparameters.space):
            hps.merge([hp])
            if hps.is_active(hp) and not isinstance(hp, hp_module.Fixed):
                if (
                    np.random.default_rng(self.seed + i).random()
                    < self.harmony_memory_accepting_rate
                ):
                    # Select a random solution from the harmony memory
                    selected_solution = harmony_memory[
                        np.random.default_rng(self.seed + i).integers(
                            self.population_size,
                        )
                    ]
                    new_value = hp.value_to_prob(selected_solution[hp.name])
                    if (
                        np.random.default_rng(self.seed + i + 1).random()
                        < self.pitch_adjusting_rate
                    ):
                        new_value += self.bandwidth * np.random.default_rng(
                            self.seed,
                        ).uniform(0, 1)
                else:
                    # Probabilities are between 0 and 1
                    new_value = np.random.uniform(0, 1)
                hps.values[hp.name] = hp.prob_to_value(np.clip(new_value, 0, 1))

        current_values = hps.values
        return current_values, previous_values

    def _update_fitness_and_best_values(self, previous_values: Hyperparams) -> None:
        """Update the fitness values and best hyperparameter values.

        Args:
            previous_values (Hyperparams): The previous hyperparameter values.
        """
        previous_score = self.trials[self.start_order[-1]].score
        worst_idx = np.argmax(self.fitness)

        if previous_score < self.fitness[worst_idx]:
            self.population[worst_idx] = previous_values.copy()
            self.fitness[worst_idx] = previous_score
            if previous_score < self.best_fitness:
                self.best_fitness = previous_score
                self.best_values = previous_values.copy()

    def get_state(self) -> dict:
        """Get the state of the HarmonySearchOracle.

        Returns:
            dict: The state of the HarmonySearchOracle.
        """
        state = super().get_state()
        state.update(
            {
                "trials_size": self.trials_size,
                "population_size": self.population_size,
                "harmony_memory_accepting_rate": self.harmony_memory_accepting_rate,
                "pitch_adjusting_rate": self.pitch_adjusting_rate,
                "bandwidth": self.bandwidth,
                "fitness": self.fitness.tolist(),
                "best_fitness": float(self.best_fitness),
                "best_values": self.best_values,
                "init_seed": self.init_seed,
                "finial_seed": self.seed,
            },
        )
        return state

    def set_state(self, state: dict) -> None:
        """Set the state of the HarmonySearchOracle.

        Args:
            state (dict): The state of the HarmonySearchOracle.
        """
        super().set_state(state)
        self.trials_size = state["trials_size"]
        self.population_size = state["population_size"]
        self.harmony_memory_accepting_rate = state["harmony_memory_accepting_rate"]
        self.pitch_adjusting_rate = state["pitch_adjusting_rate"]
        self.bandwidth = state["bandwidth"]
        self.fitness = state["fitness"]
        self.best_fitness = state["best_fitness"]
        self.best_values = state["best_values"]
        self.init_seed = state["init_seed"]
        self.finial_seed = state["finial_seed"]


@keras_tuner_export(
    [
        "keras_tuner.HarmonySearch",
        "keras_tuner.tuners.HarmonySearch",
    ],
)
class HarmonySearch(tuner_module.Tuner):
    """Tuner using Harmony Search algorithm.

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
        trials_size: The number of trials to run.
        population_size: The size of the population.
        harmony_memory_accepting_rate: The rate at which the harmony memory is accepted.
        pitch_adjusting_rate: The rate at which the pitch is adjusted.
        bandwidth: The bandwidth for adjusting the pitch.
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
        trials_size=100,
        population_size=20,
        harmony_memory_accepting_rate=0.95,
        pitch_adjusting_rate=0.7,
        bandwidth=0.01,
        seed=None,
        hyperparameters=None,
        tune_new_entries=True,
        allow_new_entries=True,
        max_retries_per_trial=0,
        max_consecutive_failed_trials=3,
        **kwargs,
    ) -> None:
        oracle = HarmonySearchOracle(
            objective=objective,
            trials_size=trials_size,
            population_size=population_size,
            harmony_memory_accepting_rate=harmony_memory_accepting_rate,
            pitch_adjusting_rate=pitch_adjusting_rate,
            bandwidth=bandwidth,
            hyperparameters=hyperparameters,
            allow_new_entries=allow_new_entries,
            tune_new_entries=tune_new_entries,
            seed=seed,
            max_retries_per_trial=max_retries_per_trial,
            max_consecutive_failed_trials=max_consecutive_failed_trials,
        )
        super().__init__(oracle=oracle, hypermodel=hypermodel, **kwargs)
