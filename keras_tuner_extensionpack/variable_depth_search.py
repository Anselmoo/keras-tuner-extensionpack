"""Variable Depth Search Oracle and Tuner."""

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


@keras_tuner_export("keras_tuner.oracles.VariableDepthSearchOracle")
class VariableDepthSearchOracle(oracle_module.Oracle):
    """Variable Depth Search Oracle.


    This oracle extends the `Oracle` class from the `keras-tuner` library and
    implements a variable depth search algorithm for hyperparameter optimization.
    It generates a population of hyperparameter sets and explores different depths
    within each set to find the best combination of hyperparameters.

    Args:
        objective: A string, `keras_tuner.Objective` instance, or a list of
            `keras_tuner.Objective`s and strings. If a string, the direction of
            the optimization (min or max) will be inferred. If a list of
            `keras_tuner.Objective`, we will minimize the sum of all the
            objectives to minimize subtracting the sum of all the objectives to
            maximize. The `objective` argument is optional when
            `Tuner.run_trial()` or `HyperModel.fit()` returns a single float as
            the objective to minimize.
        trials_size (Optional[int]): The number of trials to run.
        population_size (Optional[int]): The size of the population.
        max_depth (int): The maximum depth to explore within each hyperparameter set.
        hyperparameters (Optional[HyperParameters]): The hyperparameters to tune.
        allow_new_entries (bool): Whether to allow new hyperparameters to be added.
        tune_new_entries (bool): Whether to tune new hyperparameters.
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
        trials_size (Optional[int]): The number of trials to run.
        population_size (Optional[int]): The size of the population.
        max_depth (int): The maximum depth to explore within each hyperparameter set.
        population (List[dict]): The population of hyperparameter sets.
        fitness (numpy.ndarray): The fitness values for each hyperparameter set.
        best_fitness (float): The best fitness value found so far.
        best_values (dict): The best hyperparameter values found so far.
        init_seed (Optional[int]): The initial seed for random number generation.
        finial_seed (Optional[int]): The final seed for random number generation.

    Methods:
        populate_space(trial_id: str) -> dict:
            Fill the hyperparameter space with values for a given trial.
        _inner_loop(j, depth) -> Tuple[HP, HP]:
            Perform the inner loop of the variable depth search algorithm.
        _update_fitness_and_best_values(j, previous_values):
            Update the fitness values and best hyperparameter values.
        get_state() -> dict:
            Get the state of the oracle.
        set_state(state: dict):
            Set the state of the oracle.
    """

    def __init__(
        self,
        objective=None,
        trials_size=None,
        population_size=None,
        max_depth=50,
        hyperparameters=None,
        allow_new_entries=True,
        tune_new_entries=True,
        seed=None,
        max_retries_per_trial=0,
        max_consecutive_failed_trials=3,
    ):
        self.trials_size = trials_size
        self.population_size = population_size
        self.max_depth = max_depth
        max_trials = self.trials_size * self.population_size * (self.max_depth + 1) + 1

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
        if int(trial_id) == 1:
            # Initialize the hyperparameters set with random values
            # according to the population size.
            self.population = np.array(
                [self._random_values() for _ in range(self.population_size)],
            )
        j = int(trial_id) // (self.max_depth + 1) % self.population_size
        depth = int(trial_id) % (self.max_depth + 1)
        if int(trial_id) >= 1:
            current_values, previous_values = self._inner_loop(j, depth)
        if int(trial_id) > 0:
            self._update_fitness_and_best_values(j, previous_values)
        # Update the hyperparameters set with new values.
        if self.best_values is None:
            return {"status": trial_module.TrialStatus.STOPPED, "values": None}
        # Return the current trial's hyperparameters.
        return {"status": trial_module.TrialStatus.RUNNING, "values": current_values}

    def _inner_loop(self, j, depth) -> Tuple[HP, HP]:
        current_values = self.population[j].copy()
        previous_values = current_values.copy()
        norm_depth = depth / self.max_depth
        random_values = np.random.default_rng(self.seed).uniform(
            -norm_depth,
            +norm_depth,
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
        if previous_score < self.best_fitness:
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
                "max_depth": self.max_depth,
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
        self.max_depth = state["max_depth"]
        self.fitness = state["fitness"]
        self.best_fitness = state["best_fitness"]
        self.best_values = state["best_values"]
        self.init_seed = state["init_seed"]
        self.finial_seed = state["finial_seed"]


@keras_tuner_export(
    [
        "keras_tuner.VariableDepthSearch",
        "keras_tuner.tuners.VariableDepthSearch",
    ],
)
class VariableDepthSearch(tuner_module.Tuner):
    """Variable Depth Search Tuner.

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
        trials_size (int): The number of trials to be generated in each iteration.
        population_size (int): The number of individuals in the population.
        max_depth (int): The maximum depth of the search space.
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
        max_depth=50,
        hyperparameters=None,
        allow_new_entries=True,
        tune_new_entries=True,
        seed=None,
        max_retries_per_trial=0,
        max_consecutive_failed_trials=3,
        **kwargs,
    ):
        oracle = VariableDepthSearchOracle(
            objective=objective,
            trials_size=trials_size,
            population_size=population_size,
            max_depth=max_depth,
            hyperparameters=hyperparameters,
            allow_new_entries=allow_new_entries,
            tune_new_entries=tune_new_entries,
            seed=seed,
            max_retries_per_trial=max_retries_per_trial,
            max_consecutive_failed_trials=max_consecutive_failed_trials,
        )
        super().__init__(oracle=oracle, hypermodel=hypermodel, **kwargs)
