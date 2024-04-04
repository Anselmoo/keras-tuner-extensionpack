"""Simulated Annealing Search Tuner and Oracle."""

from __future__ import annotations

from typing import TypeVar

import numpy as np

from keras_tuner.engine import oracle as oracle_module
from keras_tuner.engine import trial as trial_module
from keras_tuner.engine import tuner as tuner_module
from keras_tuner.src.api_export import keras_tuner_export
from keras_tuner.src.engine import hyperparameters as hp_module

from keras_tuner_extensionpack.tools import exception_checks


HP = TypeVar("HP", bound=hp_module.HyperParameters)


@keras_tuner_export("keras_tuner.oracles.SimulatedAnnealingSearchOracle")
class SimulatedAnnealingSearchOracle(oracle_module.Oracle):
    """Simulated Annealing Search Oracle.

    Args:
        objective: A string, `keras_tuner.Objective` instance, or a list of
            `keras_tuner.Objective`s and strings. If a string, the direction of
            the optimization (min or max) will be inferred. If a list of
            `keras_tuner.Objective`, we will minimize the sum of all the
            objectives to minimize subtracting the sum of all the objectives to
            maximize. The `objective` argument is optional when
            `Tuner.run_trial()` or `HyperModel.fit()` returns a single float as
            the objective to minimize.
        trials_size (Optional[int]): The number of trials to run in each iteration of the algorithm.
        population_size (Optional[int]): The size of the population in each iteration of the algorithm.
        init_temperature (Optional[float]): The initial temperature for the Simulated Annealing algorithm.
        stopping_temperature (float): The stopping temperature for the Simulated Annealing algorithm.
        cooling_rate (float): The cooling rate for the Simulated Annealing algorithm.
        dynamic_cooling (bool): Whether to use dynamic cooling in the Simulated Annealing algorithm.
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
        trials_size (int): The number of trials to run in each iteration of the algorithm.
        population_size (int): The size of the population in each iteration of the algorithm.
        temperature (float): The current temperature in the Simulated Annealing algorithm.
        init_temperature (float): The initial temperature for the Simulated Annealing algorithm.
        stopping_temperature (float): The stopping temperature for the Simulated Annealing algorithm.
        cooling_rate (float): The cooling rate for the Simulated Annealing algorithm.
        dynamic_cooling (bool): Whether to use dynamic cooling in the Simulated Annealing algorithm.
        population (List[dict]): The population of hyperparameter sets.
        fitness (ndarray): The fitness values of the population.
        best_fitness (float): The best fitness value found so far.
        best_values (dict): The best set of hyperparameters found so far.
        init_seed (Optional[int]): The initial random seed used for the search.
        finial_seed (Optional[int]): The final random seed used for the search.

    Methods:
        populate_space(trial_id: str) -> dict:
            Fill the hyperparameter space with values for a given trial.
        _inner_loop(j, current_values) -> dict:
            Perform the inner loop of the Simulated Annealing algorithm.
        _delta_compare(j, current_values, previous_values) -> dict:
            Compare the current and previous solutions and decide whether to accept the new solution.
        _update_fitness_and_best_values(trial_id) -> None:
            Update the fitness and best values based on the current trial.
        get_state() -> dict:
            Get the state of the oracle.
        set_state(state: dict) -> None:
            Set the state of the oracle.
    """

    def __init__(
        self,
        objective=None,
        trials_size=None,
        population_size=None,
        init_temperature=None,
        stopping_temperature=1e-8,
        cooling_rate: float = 0.95,
        dynamic_cooling: bool = True,
        hyperparameters=None,
        allow_new_entries=True,
        tune_new_entries=True,
        seed=None,
        max_retries_per_trial=0,
        max_consecutive_failed_trials=3,
    ) -> None:
        self.trials_size = trials_size
        self.population_size = population_size

        exception_checks.not_in_range_check(
            "cooling_rate",
            cooling_rate,
        )

        self.temperature = self.init_temperature = (
            init_temperature if init_temperature else 9659.243
        )
        self.stopping_temperature = stopping_temperature

        self.cooling_rate = cooling_rate
        self.dynamic_cooling = dynamic_cooling

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
        self.population = [{}]
        self.fitness = np.empty(self.population_size)
        self.best_fitness = np.inf
        self.best_values = {}
        self.init_seed = seed
        self.finial_seed = seed

    def populate_space(self, trial_id: str) -> dict:
        """Fill the hyperparameter space with values."""
        if int(trial_id) == 0:
            self.init_seed = self.seed
            return {"status": trial_module.TrialStatus.RUNNING, "values": {}}
        # Change the seed for each trial.
        self.seed += 1
        if int(trial_id) in {1, 2}:
            # Initialize the hyperparameters set with random values
            # according to the population size.
            self.population.append(self._random_values())

            return {
                "status": trial_module.TrialStatus.RUNNING,
                "values": self.population[0],
            }

        j = int(trial_id) % self.trials_size
        if j == 0:
            self.temperature = self.init_temperature
            self._update_fitness_and_best_values(trial_id)
            self.population.append(self._random_values())
        current_values = self._inner_loop(j, self.population[-1])
        self.population.append(current_values)
        # self.population[j] = current_values
        # self._update_fitness_and_best_values(trial_id)
        self._delta_compare(
            j, current_values=self.population[-1], previous_values=self.population[-2]
        )

        if self.best_values is None:
            return {"status": trial_module.TrialStatus.STOPPED, "values": None}
        return {
            "status": trial_module.TrialStatus.RUNNING,
            "values": current_values,
        }

    def _inner_loop(self, j, current_values) -> dict:
        # dim = len(self.hyperparameters.space)

        hps = hp_module.HyperParameters()
        for i, hp in enumerate(self.hyperparameters.space):
            hps.merge([hp])
            if hps.is_active(hp) and not isinstance(hp, hp_module.Fixed):
                new_value = hp.value_to_prob(
                    current_values[hp.name]
                ) + np.random.default_rng(self.seed + i + j).uniform(-0.5, 0.5)
                new_value = np.clip(new_value, 0, 1)
                hps.values[hp.name] = hp.prob_to_value(new_value)

        return hps.values

    def _delta_compare(self, j, current_values, previous_values):
        hps = hp_module.HyperParameters()
        # Calculate the change in cost
        delta_cost = (
            self.trials[self.start_order[-1]].score
            - self.trials[self.start_order[-2]].score
            if j > 0
            else self.trials[self.start_order[-1]].score
        )

        # Decide whether to accept the new solution
        if delta_cost < 0 or np.random.default_rng(self.seed).random() < np.exp(
            -delta_cost / self.temperature
        ):
            current_values = previous_values

        # Update the temperature
        if self.dynamic_cooling:
            self.temperature *= self.cooling_rate
            print(">>>", self.temperature)

        # Reset the solution and temperature if the temperature is too low
        if self.temperature < self.stopping_temperature:
            for i, hp in enumerate(self.hyperparameters.space):
                hps.merge([hp])
                if hps.is_active(hp) and not isinstance(hp, hp_module.Fixed):
                    new_value = np.random.default_rng(self.seed).uniform(0, 1)
                    hps.values[hp.name] = hp.prob_to_value(new_value)
            current_values = hps.values
            self.temperature = self.init_temperature

        return current_values

    def _update_fitness_and_best_values(self, trial_id) -> None:
        # Update the fitness of the previous trial.
        if int(trial_id) > 0:
            self.fitness[(int(trial_id) - 1) % self.trials_size] = self.trials[
                self.start_order[-1]
            ].score

        # Update the best fitness and values.
        if self.fitness[(int(trial_id) - 1) % self.trials_size] < self.best_fitness:
            self.best_fitness = self.fitness[(int(trial_id) - 1) % self.trials_size]
            self.best_values = self.population[(int(trial_id) - 1) % self.trials_size]

    def get_state(self):
        state = super().get_state()
        state.update(
            {
                "trials_size": self.trials_size,
                "population_size": self.population_size,
                "init_temperature": self.init_temperature,
                "stopping_temperature": self.stopping_temperature,
                "cooling_rate": self.cooling_rate,
                "dynamic_cooling": self.dynamic_cooling,
                "fitness": self.fitness.tolist(),
                "best_fitness": float(self.best_fitness),
                "best_values": self.best_values,
                "init_seed": self.init_seed,
                "finial_seed": self.seed,
            },
        )
        return state

    def set_state(self, state) -> None:
        super().set_state(state)
        self.trials_size = state["trials_size"]
        self.population_size = state["population_size"]
        self.init_temperature = state["init_temperature"]
        self.stopping_temperature = state["stopping_temperature"]
        self.cooling_rate = state["cooling_rate"]
        self.dynamic_cooling = state["dynamic_cooling"]
        self.fitness = state["fitness"]
        self.best_fitness = state["best_fitness"]
        self.best_values = state["best_values"]
        self.init_seed = state["init_seed"]
        self.finial_seed = state["finial_seed"]


@keras_tuner_export(
    [
        "keras_tuner.SimulatedAnnealingSearch",
        "keras_tuner.tuners.SimulatedAnnealingSearch",
    ],
)
class SimulatedAnnealingSearch(tuner_module.Tuner):
    """Simulated Annealing Search Tuner.

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
        trials_size (Optional[int]): The number of trials to run in each iteration
            of the algorithm.
        population_size (Optional[int]): The size of the population in each iteration
            of the algorithm.
        init_temperature (Optional[float]): The initial temperature for the Simulated
            Annealing algorithm.
        stopping_temperature (float): The stopping temperature for the Simulated
            Annealing algorithm.
        cooling_rate (float): The cooling rate for the Simulated Annealing algorithm.
        dynamic_cooling (bool): Whether to use dynamic cooling in the Simulated
            Annealing algorithm.
        hyperparameters: Optional `HyperParameters` instance. Can be used to
            override (or register in advance) hyperparameters in the search
            space.
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
        init_temperature=None,
        stopping_temperature=1e-8,
        cooling_rate: float = 0.95,
        dynamic_cooling: bool = True,
        seed=None,
        hyperparameters=None,
        tune_new_entries=True,
        allow_new_entries=True,
        max_retries_per_trial=0,
        max_consecutive_failed_trials=3,
        **kwargs,
    ) -> None:
        oracle = SimulatedAnnealingSearchOracle(
            objective=objective,
            trials_size=trials_size,
            population_size=population_size,
            init_temperature=init_temperature,
            stopping_temperature=stopping_temperature,
            cooling_rate=cooling_rate,
            dynamic_cooling=dynamic_cooling,
            hyperparameters=hyperparameters,
            allow_new_entries=allow_new_entries,
            tune_new_entries=tune_new_entries,
            seed=seed,
            max_retries_per_trial=max_retries_per_trial,
            max_consecutive_failed_trials=max_consecutive_failed_trials,
        )
        super().__init__(oracle=oracle, hypermodel=hypermodel, **kwargs)
