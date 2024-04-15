import keras_tuner
import keras
from keras_tuner.engine import hyperparameters as hp_module
import pytest

from keras_tuner_extensionpack import sin_cosine_algorithm as sca_module
from keras_tuner_extensionpack import tabu_search as ts_module
from keras_tuner_extensionpack import variable_depth_search as vds_module
from keras_tuner_extensionpack import harmony_search as hs_module
from keras_tuner_extensionpack import simulated_annealing_search as sa_module
from keras_tuner_extensionpack import differential_evolution as de_module
from keras_tuner_extensionpack import cma_es_algorithm as cma_module
from benchmark.functions import shifted_ackley


def build_model(hp):
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(2, 2)))
    for i in range(3):
        model.add(
            keras.layers.Dense(
                units=hp.Int(f"units_{str(i)}", 2, 4, 2), activation="relu"
            )
        )


@pytest.mark.parametrize(
    "oracle_class",
    [
        sca_module.SinCosineAlgorithmOracle,
        ts_module.TabuSearchOracle,
        vds_module.VariableDepthSearchOracle,
        hs_module.HarmonySearchOracle,
        sa_module.SimulatedAnnealingSearchOracle,
        de_module.DifferentialEvolutionOracle,
        cma_module.CMAESAlgorithmOracle,
    ],
)
def test_oracle(tmp_path, oracle_class):
    hps = hp_module.HyperParameters()
    hps.Choice("a", [1, 2], default=1)
    hps.Int("b", 3, 10, default=3)
    hps.Float("c", 0, 1, 0.1, default=0)
    hps.Fixed("d", 7)
    hps.Choice("e", [9, 0], default=9)
    oracle = oracle_class(
        objective=keras_tuner.Objective("score", "max"),
        trials_size=20,
        population_size=10,
        hyperparameters=hps,
    )
    oracle._set_project_dir(tmp_path, "untitled")
    for i in range(5):
        trial = oracle.create_trial(str(i))
        oracle.update_trial(trial.trial_id, {"score": i})
        trial.status = "COMPLETED"
        oracle.end_trial(trial)


@pytest.mark.parametrize(
    "oracle_class",
    [
        sca_module.SinCosineAlgorithmOracle,
        ts_module.TabuSearchOracle,
        vds_module.VariableDepthSearchOracle,
        hs_module.HarmonySearchOracle,
        sa_module.SimulatedAnnealingSearchOracle,
        de_module.DifferentialEvolutionOracle,
        cma_module.CMAESAlgorithmOracle,
    ],
)
def test_oracle_with_zero_y(tmp_path, oracle_class):
    hps = hp_module.HyperParameters()
    hps.Choice("a", [1, 2], default=1)
    hps.Int("b", 3, 10, default=3)
    hps.Float("c", 0, 1, 0.1, default=0)
    hps.Fixed("d", 7)
    hps.Choice("e", [9, 0], default=9)
    oracle = oracle_class(
        objective=keras_tuner.Objective("score", "max"),
        trials_size=20,
        population_size=10,
        hyperparameters=hps,
    )
    oracle._set_project_dir(tmp_path, "untitled")
    for i in range(5):
        trial = oracle.create_trial(str(i))
        oracle.update_trial(trial.trial_id, {"score": 0})
        trial.status = "COMPLETED"
        oracle.end_trial(trial)


@pytest.mark.parametrize(
    "oracle_class",
    [
        [sca_module.SinCosineAlgorithm, sca_module.SinCosineAlgorithmOracle],
        [ts_module.TabuSearch, ts_module.TabuSearchOracle],
        [vds_module.VariableDepthSearch, vds_module.VariableDepthSearchOracle],
        [hs_module.HarmonySearch, hs_module.HarmonySearchOracle],
        [sa_module.SimulatedAnnealingSearch, sa_module.SimulatedAnnealingSearchOracle],
        [de_module.DifferentialEvolution, de_module.DifferentialEvolutionOracle],
        [cma_module.CMAESAlgorithm, cma_module.CMAESAlgorithmOracle],
    ],
)
def test_sincos_optimization_tuner(tmp_path, oracle_class):
    tuner = oracle_class[0](
        build_model,
        objective="val_accuracy",
        trials_size=20,
        population_size=10,
        directory=tmp_path,
    )
    assert isinstance(tuner.oracle, oracle_class[1])


def test_sincos_optimization_tuner_set_alpha_beta(tmp_path):
    tuner = sca_module.SinCosineAlgorithm(
        build_model,
        objective="val_accuracy",
        trials_size=20,
        population_size=10,
        directory=tmp_path,
        r1_cut=0.5,
        r2_cut=0.5,
    )
    assert isinstance(tuner.oracle, sca_module.SinCosineAlgorithmOracle)


class MyTuner(sca_module.SinCosineAlgorithm):
    def run_trial(self, trial, *args, **kwargs):
        hp = trial.hyperparameters
        x = hp.Float("x", min_value=-5, max_value=5, step=0.01, sampling="linear")
        y = hp.Float("y", min_value=-5, max_value=5, step=0.01, sampling="linear")
        _ = hp.Fixed("z", value=1)
        _ = hp.Choice("c", values=["1", "2", "3"])
        return shifted_ackley([x, y])


@pytest.mark.parametrize("normalizer", ["sigmoid", "clip", "absclip", "squareclip"])
def test_my_tuner(normalizer: str) -> None:
    tuner = MyTuner(
        overwrite=True,
        directory="hyperband",
        project_name="tune_anything",
        trials_size=5,
        population_size=5,
        r1_cut=0.3,
        r2_cut=0.3,
        amplitude=1,
        damping_factor=1,
        new_value_normalizer=normalizer,
        objective=keras_tuner.Objective("roc", direction="min"),
    )

    tuner.search()

    best_hp = tuner.get_best_hyperparameters()[0]

    assert -5 <= best_hp.get("x") <= 5
    assert -5 <= best_hp.get("y") <= 5


@pytest.mark.parametrize(
    "oracle_class",
    [
        sca_module.SinCosineAlgorithm,
        ts_module.TabuSearch,
        vds_module.VariableDepthSearch,
        hs_module.HarmonySearch,
        sa_module.SimulatedAnnealingSearch,
        de_module.DifferentialEvolution,
        cma_module.CMAESAlgorithm,
    ],
)
def test_multi_tuner(oracle_class, tmp_path) -> None:
    class MultiTuner(oracle_class):
        def run_trial(self, trial, *args, **kwargs):
            hp = trial.hyperparameters
            x = hp.Float("x", min_value=-5, max_value=5, step=0.01, sampling="linear")
            y = hp.Float("y", min_value=-5, max_value=5, step=0.01, sampling="linear")
            _ = hp.Fixed("z", value=1)
            _ = hp.Choice("c", values=["1", "2", "3"])
            return shifted_ackley([x, y])

    tuner = MultiTuner(
        overwrite=True,
        directory=tmp_path,
        project_name="tune_anything",
        objective=keras_tuner.Objective("roc", direction="min"),
        trials_size=5,
        population_size=5,
    )

    tuner.search()

    best_hp = tuner.get_best_hyperparameters()[0]

    assert -5 <= best_hp.get("x") <= 5
    assert -5 <= best_hp.get("y") <= 5
