"""Utility functions for converting between hyperparameters and arrays."""

from __future__ import annotations

import numpy as np

from keras_tuner.src.engine import hyperparameters as hp_module

from keras_tuner_extensionpack.tools.type_definitions import HP


def hp2array(hyperparameters: HP, solution: HP) -> np.ndarray:
    """Convert hyperparameters and solution to a numpy array of numeric values.

    Args:
        hyperparameters (HP): The hyperparameters object containing the search space.
        solution (HP): The solution object containing the selected hyperparameter
            values.

    Returns:
        np.ndarray: A numpy array containing the numeric values of the selected
            hyperparameters.
    """
    hps = hp_module.HyperParameters()
    numeric_values = []
    for hp in hyperparameters.space:
        hps.merge([hp])
        if hps.is_active(hp):
            numeric_values.append(hp.value_to_prob(solution[hp.name]))
    return np.array(numeric_values)


def array2hp(hyperparameters: HP, array: np.ndarray) -> HP:
    """Converts an array of values to a dictionary of hyperparameter values.

    Args:
        hyperparameters (HP): The hyperparameters object defining the search space.
        array (np.ndarray): The array of values to convert.

    Returns:
        HP: A dictionary of hyperparameter values.
    """
    hps = hp_module.HyperParameters()
    for i, hp in enumerate(hyperparameters.space):
        hps.merge([hp])
        if hps.is_active(hp) and not isinstance(hp, hp_module.Fixed):
            hps.values[hp.name] = hp.prob_to_value(array[i])
    return hps.values
