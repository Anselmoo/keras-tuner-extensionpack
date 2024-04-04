"""Tabu search optimizer oracle and tuner for Keras Tuner."""

from __future__ import annotations

from typing import Tuple
from typing import TypeVar

import numpy as np

from keras_tuner.engine import oracle as oracle_module
from keras_tuner.engine import trial as trial_module
from keras_tuner.engine import tuner as tuner_module
from keras_tuner.src.api_export import keras_tuner_export
from keras_tuner.src.engine import hyperparameters as hp_module


HP = TypeVar("HP", bound=hp_module.HyperParameters)


@keras_tuner_export("keras_tuner.oracles.TabuSearchOracle")
class TabuSearchOracle(oracle_module.Oracle):
    """Tabu search optimizer oracle for Keras Tuner.

    Args:
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
        tabu_list_size: The size of the tabu list.
        perturbation_depth: The depth of perturbation for generating new solutions.
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
        trials_size: The number of trials to run.
        population_size: The size of the population.
        tabu_list_size: The size of the tabu list.
        perturbation_depth: The depth of perturbation for generating new solutions.
        population: The current population of solutions.
        fitness: The fitness values of the population.
        best_fitness: The best fitness value found so far.
        best_values: The best set of hyperparameters found so far.
        tabu_list: The tabu list of previously visited solutions.
        init_seed: The initial random seed.
        finial_seed: The final random seed.

    Methods:
        populate_space: Fill the hyperparameter space with values for a given trial.
        tabu_check: Check and update the tabu list.
        _inner_loop: Perform the inner loop of the Tabu Search algorithm.
        _update_fitness_and_best_values: Update the fitness values and best hyperparameters.
        get_state: Get the state of the oracle.
        set_state: Set the state of the oracle.
    """

    def __init__(
        self,
        objective=None,
        trials_size=None,
        population_size=None,
        tabu_list_size=50,
        perturbation_depth=0.1,
        seed=None,
        hyperparameters=None,
        tune_new_entries=True,
        allow_new_entries=True,
        max_retries_per_trial=0,
        max_consecutive_failed_trials=3,
    ):
        self.trials_size = trials_size
        self.population_size = population_size
        self.tabu_list_size = tabu_list_size
        if perturbation_depth < 0 or perturbation_depth > 1:
            msg = "Pertubation depth must be in the range [0, 1]."
            raise ValueError(msg)
        self.perturbation_depth = perturbation_depth
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
        self.tabu_list = []
        self.init_seed = seed
        self.finial_seed = seed

    def populate_space(self, trial_id: str) -> dict:
        """Fill the hyperparameter space with values."""
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
            self.best_values = self.population[0].copy()

        # Here Inner loop
        j = int(trial_id) % self.population_size

        if int(trial_id) >= 1:
            self.tabu_check()
            current_values, previous_values = self._inner_loop()
        if int(trial_id) > 0:
            self._update_fitness_and_best_values(j, previous_values)
        # Update the hyperparameters set with new values.
        if self.best_values is None:
            return {"status": trial_module.TrialStatus.STOPPED, "values": None}
        # Return the current trial's hyperparameters.
        return {"status": trial_module.TrialStatus.RUNNING, "values": current_values}

    def tabu_check(self):
        if len(self.tabu_list) >= self.tabu_list_size:
            self.tabu_list.pop(0)
        self.tabu_list.append(self.best_values)

    def _inner_loop(self) -> Tuple[HP, HP]:
        current_values = self.best_values.copy()
        previous_values = current_values.copy()
        random_values = np.random.default_rng(self.seed).uniform(
            -self.perturbation_depth,
            self.perturbation_depth,
            size=len(current_values),
        )
        hps = hp_module.HyperParameters()
        for i, hp in enumerate(self.hyperparameters.space):
            hps.merge([hp])
            if hps.is_active(hp) and not isinstance(hp, hp_module.Fixed):
                new_values = (
                    hp.value_to_prob(current_values[hp.name]) + random_values[i]
                )
                hps.values[hp.name] = hp.prob_to_value(np.clip(new_values, 0, 1))
        current_values = hps.values
        return current_values, previous_values

    def _update_fitness_and_best_values(self, j, previous_values):
        previous_score = self.trials[self.start_order[-1]].score
        if previous_score < self.best_fitness and not any(
            np.array_equal(x, previous_values) for x in self.tabu_list
        ):
            self.best_fitness = previous_score
            self.best_values = previous_values
            self.population[j] = self.best_values.copy()
        self.fitness[j] = previous_score

    def get_state(self):
        state = super().get_state()
        state.update(
            {
                "trials_size": self.trials_size,
                "population_size": self.population_size,
                "tabu_list_size": self.tabu_list_size,
                "fitness": self.fitness.tolist(),
                "best_fitness": float(self.best_fitness),
                "best_values": self.best_values,
                "init_seed": self.init_seed,
                "finial_seed": self.seed,
            },
        )
        return state

    def set_state(self, state):
        super().set_state(state)
        self.trials_size = state["trials_size"]
        self.population_size = state["population_size"]

        self.tabu_list_size = state["tabu_list_size"]
        self.fitness = state["fitness"]
        self.best_fitness = state["best_fitness"]
        self.best_values = state["best_values"]
        self.init_seed = state["init_seed"]
        self.finial_seed = state["finial_seed"]


@keras_tuner_export(
    [
        "keras_tuner.TabuSearch",
        "keras_tuner.tuners.TabuSearch",
    ],
)
class TabuSearch(tuner_module.Tuner):
    """Tabu search optimizer tuner for Keras Tuner.

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
        trials_size (Optional[int]): The number of trials to be generated in each iteration.
        population_size (Optional[int]): The number of solutions to be generated in each iteration.
        perturbation_depth (Optional[float]): The magnitude of perturbation applied to the solutions.
        tabu_list_size (Optional[int]): The size of the tabu list used to store the visited solutions.
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
        perturbation_depth=0.1,
        tabu_list_size=50,
        hyperparameters=None,
        allow_new_entries=True,
        tune_new_entries=True,
        seed=None,
        max_retries_per_trial=0,
        max_consecutive_failed_trials=3,
        **kwargs,
    ):
        oracle = TabuSearchOracle(
            objective=objective,
            trials_size=trials_size,
            population_size=population_size,
            perturbation_depth=perturbation_depth,
            tabu_list_size=tabu_list_size,
            hyperparameters=hyperparameters,
            allow_new_entries=allow_new_entries,
            tune_new_entries=tune_new_entries,
            seed=seed,
            max_retries_per_trial=max_retries_per_trial,
            max_consecutive_failed_trials=max_consecutive_failed_trials,
        )
        super().__init__(oracle=oracle, hypermodel=hypermodel, **kwargs)
