"""Example of using CMA-ES algorithm with Keras Tuner ExtensionPack."""

from benchmark.functions import shifted_ackley
from keras_tuner_extensionpack.cma_es_algorithm import CMAESAlgorithm
import keras_tuner


class MyTuner(CMAESAlgorithm):
    def run_trial(self, trial, *args, **kwargs):
        # Get the hp from trial.
        hp = trial.hyperparameters
        # Define "x" as a hyperparameter.
        x = hp.Float(
            "x",
            min_value=-5,
            max_value=5,
            step=1e-8,
            sampling="linear",
        )
        y = hp.Float(
            "y",
            min_value=-5,
            max_value=5,
            step=1e-8,
            sampling="linear",
        )
        return shifted_ackley([x, y])


tuner = MyTuner(
    # No hypermodel or objective specified.
    overwrite=True,
    directory="cma_es",
    project_name="tune_anything",
    trials_size=100,
    population_size=100,
    sigma_init=0.5,
    epsilon=1e-8,
    objective=keras_tuner.Objective("roc", direction="min"),
)

# No need to pass anything to search()
# unless you use them in run_trial().
tuner.search()
print(tuner.get_best_hyperparameters()[0].get("x"))
print(tuner.get_best_hyperparameters()[0].get("y"))
