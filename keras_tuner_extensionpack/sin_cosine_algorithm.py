"""Sin-Cosine Algorithm for Hyperparameter Optimization."""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
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
from keras_tuner_extensionpack.tools.type_definitions import Population


class SinCosineMethod:
    """Class for the Sin-Cosine Algorithm methods."""

    def generate_random_numbers(
        self,
        seed: int,
        dim: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generates random numbers using three different seeds.

        Args:
            seed (int): The seed value for random number generation.
            dim (int): The dimension of the random number arrays.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing three
                arrays of float random numbers.
        """
        r1 = np.random.default_rng(seed + 1).random(dim)
        r2 = np.random.default_rng(seed + 2).random(dim)
        r3 = np.random.default_rng(seed + 3).random(dim)
        return r1, r2, r3

    def calculate_oscillations(
        self,
        population: Population,
        best_values: Hyperparams,
        i: int,
        j: int,
        hp: HP,
        mask1: np.ndarray,
        mask2: np.ndarray,
        r3: np.ndarray,
        amplitude: float,
        damping_factor: float,
        shift_up: float = 0,
    ) -> float:
        """Calculates the oscillations for a given population and hyperparameters.

        Args:
            population (Population): The population of hyperparameters.
            best_values (Hyperparams): The best hyperparameter values.
            i (int): The index of the hyperparameter.
            j (int): The index of the population member.
            hp (HP): The hyperparameter object.
            mask1 (np.ndarray): The first mask array.
            mask2 (np.ndarray): The second mask array.
            r3 (np.ndarray): The third array.
            amplitude (float): The amplitude value.
            damping_factor (float): The damping factor value.
            shift_up (float): The shift up the center of the oscillations.

        Returns:
            float: The new value after calculating the oscillations.
        """
        if mask1[i] and mask2[i]:
            new_value = hp.value_to_prob(population[j][hp.name]) + amplitude * np.sin(
                (r3[i] * damping_factor) + shift_up,
            ) * np.abs(
                r3[i] * hp.value_to_prob(best_values[hp.name])
                - hp.value_to_prob(population[j][hp.name]),
            )
        elif mask1[i] and not mask2[i]:
            new_value = hp.value_to_prob(population[j][hp.name]) + amplitude * np.cos(
                (r3[i] * damping_factor) + shift_up,
            ) * np.abs(
                r3[i] * hp.value_to_prob(best_values[hp.name])
                - hp.value_to_prob(population[j][hp.name]),
            )
        elif not mask1[i] and mask2[i]:
            new_value = hp.value_to_prob(population[j][hp.name]) - amplitude * np.sin(
                (r3[i] * damping_factor) + shift_up,
            ) * np.abs(
                r3[i] * hp.value_to_prob(best_values[hp.name])
                - hp.value_to_prob(population[j][hp.name]),
            )
        else:
            new_value = hp.value_to_prob(population[j][hp.name]) - amplitude * np.cos(
                (r3[i] * damping_factor) + shift_up,
            ) * np.abs(
                r3[i] * hp.value_to_prob(best_values[hp.name])
                - hp.value_to_prob(population[j][hp.name]),
            )
        return new_value

    def clip_new_value(self, new_value: float) -> float:
        """Clips the new value to the range [0, 1].

        Args:
            new_value (float): The new value to clip.

        Returns:
            float: The clipped new value.
        """
        return np.clip(new_value, 0, 1)

    def absclip_new_value(self, new_value: float) -> float:
        """Clips the new absolute value to the range [0, 1].

        Args:
            new_value (float): The new value to clip.

        Returns:
            float: The clipped new value.
        """
        return self.clip_new_value(np.abs(new_value))

    def squareclip_new_value(self, new_value: float) -> float:
        """Clips the new square value to the range [0, 1].

        Args:
            new_value (float): The new value to clip.

        Returns:
            float: The clipped new value.
        """
        return self.clip_new_value(np.square(new_value))

    def sigmoid_new_value(self, new_value: float) -> float:
        """Applies the sigmoid function to the new value.

        Args:
            new_value (float): The new value to normalize.

        Returns:
            float: The normalized new value.
        """
        return 1 / (1 + np.exp(-2 * np.clip(new_value, -1, 1)))


@keras_tuner_export("keras_tuner.oracles.SinCosineAlgorithmOracle")
class SinCosineAlgorithmOracle(oracle_module.Oracle):
    """Oracle implementation for the Sin-Cosine Algorithm.

    Args:
        objective: A string, `keras_tuner.Objective` instance, or a list of
            `keras_tuner.Objective`s and strings. If a string, the direction of
            the optimization (min or max) will be inferred. If a list of
            `keras_tuner.Objective`, we will minimize the sum of all the
            objectives to minimize subtracting the sum of all the objectives to
            maximize. The `objective` argument is optional when
            `Tuner.run_trial()` or `HyperModel.fit()` returns a single float as
            the objective to minimize.
        trials_size (Optional[int]): The number of trials to run in each generation.
        population_size (Optional[int]): The size of the population in each generation.
        r1_cut (float): The cutoff value for the first random number.
        r2_cut (float): The cutoff value for the second random number.
        amplitude (float): The amplitude of the oscillations.
        damping_factor (float): The damping factor for the oscillations.
        shift_up (float): The shift up value for the oscillations.
        new_value_normalizer (str): The method to normalize the new values.
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
        trials_size (int): The number of trials to run in each generation.
        population_size (int): The size of the population in each generation.
        r1_cut (float): The cutoff value for the first random number.
        r2_cut (float): The cutoff value for the second random number.
        amplitude (float): The amplitude of the oscillations.
        damping_factor (float): The damping factor for the oscillations.
        shift_up (float): The shift up value for the oscillations.
        new_value_normalizer (str): The method to normalize the new values.
        hyperparameters (HyperParameters): The hyperparameters to tune.
        allow_new_entries (bool): Whether to allow new hyperparameter entries.
        tune_new_entries (bool): Whether to tune new hyperparameter entries.
        seed (int): The seed value for random number generation.
        max_retries_per_trial (int): The maximum number of retries per trial.
        max_consecutive_failed_trials (int): The maximum number of consecutive failed trials.
        population (List[dict]): The population of hyperparameter sets.
        fitness (ndarray): The fitness values of the population.
        best_fitness (float): The best fitness value.
        best_values (dict): The best set of hyperparameters.
        tabu_list (List): The tabu list for preventing duplicate trials.
        init_seed (int): The initial seed value.
        finial_seed (int): The final seed value.
        method (SinCosineMethod): The SinCosineMethod instance for calculating oscillations.

    Methods:
        initialize_clip_methods(new_value_normalizer: str) -> None:
            Initialize the clip methods based on the new value normalizer.
        populate_space(trial_id: str) -> dict:
            Fill the hyperparameter space with values for a given trial.
        _inner_loop(j) -> dict:
            Perform the inner loop of the Sin-Cosine Algorithm.
        _update_fitness_and_best_values(trial_id) -> None:
            Update the fitness values and best hyperparameters.
        get_state() -> dict:
            Get the state of the SinCosineAlgorithmOracle.
        set_state(state: dict) -> None:
            Set the state of the SinCosineAlgorithmOracle.
    """

    def __init__(
        self,
        objective=None,
        trials_size=None,
        population_size=None,
        r1_cut=0.5,
        r2_cut=0.5,
        amplitude=1,
        damping_factor=1,
        shift_up=0,
        new_value_normalizer="clip",
        hyperparameters=None,
        allow_new_entries=True,
        tune_new_entries=True,
        seed=None,
        max_retries_per_trial=0,
        max_consecutive_failed_trials=3,
    ):
        self.trials_size = trials_size
        self.population_size = population_size

        exception_checks.not_in_range_check("r1_cut", r1_cut)
        exception_checks.not_in_range_check("r2_cut", r2_cut)
        exception_checks.not_in_range_check(
            "amplitude",
            amplitude,
            upper_bound_include=True,
        )
        exception_checks.not_in_range_check(
            "damping_factor",
            damping_factor,
            upper_bound_include=True,
        )
        exception_checks.not_in_range_check(
            "shift_up",
            shift_up,
            lower_bound=-1,
            upper_bound=1,
            lower_bound_include=True,
            upper_bound_include=True,
        )
        self.r1_cut = r1_cut
        self.r2_cut = r2_cut
        self.amplitude = amplitude
        self.damping_factor = damping_factor
        self.shift_up = shift_up

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
        self.method = SinCosineMethod()
        self.method.new_value_normalizer = None
        self.initialize_clip_methods(new_value_normalizer)

    def initialize_clip_methods(self, new_value_normalizer: str) -> None:
        if new_value_normalizer.lower() == "sigmoid":
            self.method.new_value_normalizer = self.method.sigmoid_new_value
        elif new_value_normalizer.lower() == "clip":
            self.method.new_value_normalizer = self.method.clip_new_value
        elif new_value_normalizer.lower() == "absclip":
            self.method.new_value_normalizer = self.method.absclip_new_value
        elif new_value_normalizer.lower() == "squareclip":
            self.method.new_value_normalizer = self.method.squareclip_new_value
        else:
            msg = (
                "New value normalizer must be one of 'sigmoid', "
                "'clip', 'absclip', 'squareclip'."
            )
            raise ValueError(msg)

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
        j = int(trial_id) % self.population_size
        if j == 0:
            best_idx = np.argmin(self.fitness)
            self.best_values = self.population[best_idx]
            self.best_fitness = self.fitness[best_idx]
        current_values = self._inner_loop(j)
        self.population[j] = current_values
        self._update_fitness_and_best_values(trial_id)

        if self.best_values is None:
            return {"status": trial_module.TrialStatus.STOPPED, "values": None}
        return {
            "status": trial_module.TrialStatus.RUNNING,
            "values": current_values,
        }

    def _inner_loop(self, j) -> dict:
        dim = len(self.hyperparameters.space)

        # Generate random numbers
        r1, r2, r3 = self.method.generate_random_numbers(self.seed, dim)

        # Update position using vectorized operations
        mask1 = r1 < self.r1_cut
        mask2 = r2 < self.r2_cut

        hps = hp_module.HyperParameters()
        for i, hp in enumerate(self.hyperparameters.space):
            hps.merge([hp])
            if hps.is_active(hp) and not isinstance(hp, hp_module.Fixed):
                new_value = self.method.calculate_oscillations(
                    population=self.population,
                    best_values=self.best_values,
                    i=i,
                    j=j,
                    hp=hp,
                    mask1=mask1,
                    mask2=mask2,
                    r3=r3,
                    amplitude=self.amplitude,
                    damping_factor=self.damping_factor,
                    shift_up=self.shift_up,
                )
                new_value = self.method.new_value_normalizer(new_value)
                hps.values[hp.name] = hp.prob_to_value(new_value)

        return hps.values

    def _update_fitness_and_best_values(self, trial_id) -> None:
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

    def get_state(self) -> dict:
        state = super().get_state()
        state.update(
            {
                "trials_size": self.trials_size,
                "population_size": self.population_size,
                "r1_cut": self.r1_cut,
                "r2_cut": self.r2_cut,
                "amplitude": self.amplitude,
                "damping_factor": self.damping_factor,
                "shift_up": self.shift_up,
                "fitness": self.fitness.tolist(),
                "best_fitness": float(self.best_fitness),
                "best_values": self.best_values,
                "init_seed": self.init_seed,
                "finial_seed": self.seed,
            },
        )
        return state

    def set_state(self, state: dict) -> None:
        super().set_state(state)
        self.trials_size = state["trials_size"]
        self.population_size = state["population_size"]
        self.r1_cut = state["r1_cut"]
        self.r2_cut = state["r2_cut"]
        self.amplitude = (state["amplitude"],)
        self.damping_factor = (state["damping_factor"],)
        self.shift_up = (state["shift_up"],)
        self.fitness = state["fitness"]
        self.best_fitness = state["best_fitness"]
        self.best_values = state["best_values"]
        self.init_seed = state["init_seed"]
        self.finial_seed = state["finial_seed"]


@keras_tuner_export(
    [
        "keras_tuner.SinCosineAlgorithm",
        "keras_tuner.tuners.SinCosineAlgorithm",
    ],
)
class SinCosineAlgorithm(tuner_module.Tuner):
    """SinCosine algorithm for hyperparameter optimization.

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
        trials_size: The number of trials to be generated in each iteration of the algorithm.
        population_size: The number of individuals in the population.
        r1_cut: The probability of applying the first crossover operator.
        r2_cut: The probability of applying the second crossover operator.
        amplitude: The amplitude of the sinusoidal function used for mutation.
        shift_up: The shift value for the sinusoidal function used for mutation.
        new_value_normalizer: The normalization function for new values generated during mutation.
        damping_factor: The damping factor for the sinusoidal function used for mutation.
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
        r1_cut=0.5,
        r2_cut=0.5,
        amplitude=1,
        shift_up=0,
        new_value_normalizer="sigmoid",
        damping_factor=1,
        hyperparameters=None,
        allow_new_entries=True,
        tune_new_entries=True,
        seed=None,
        max_retries_per_trial=0,
        max_consecutive_failed_trials=3,
        **kwargs,
    ):
        oracle = SinCosineAlgorithmOracle(
            objective=objective,
            trials_size=trials_size,
            population_size=population_size,
            r1_cut=r1_cut,
            r2_cut=r2_cut,
            amplitude=amplitude,
            new_value_normalizer=new_value_normalizer,
            damping_factor=damping_factor,
            shift_up=shift_up,
            hyperparameters=hyperparameters,
            allow_new_entries=allow_new_entries,
            tune_new_entries=tune_new_entries,
            seed=seed,
            max_retries_per_trial=max_retries_per_trial,
            max_consecutive_failed_trials=max_consecutive_failed_trials,
        )
        super().__init__(oracle=oracle, hypermodel=hypermodel, **kwargs)
