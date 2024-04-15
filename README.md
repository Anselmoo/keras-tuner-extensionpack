# keras-tuner-extensionpack

An extension package for [KerasTuner](https://github.com/keras-team/keras-tuner) for providing more optimizers.

## Currently Implemented Algorithms

This package extends KerasTuner with additional optimization algorithms. The currently implemented tuners include:

- **CMA-ES (Covariance Matrix Adaptation Evolution Strategy)**: An evolutionary algorithm for difficult non-linear non-convex optimization problems in continuous domain.

- **Differential Evolution**: A metaheuristic approach that is useful for global optimization of a multidimensional function.

- **Harmony Search**: A music-inspired algorithm that is based on the improvisation process of musicians.

- **Simulated Annealing**: A probabilistic technique for approximating the global optimum of a given function.

- **Sine Cosine Algorithm**: An optimization algorithm inspired by the sine and cosine mathematical functions.

- **Tabu Search**: A metaheuristic search method using local or neighborhood search procedures for mathematical optimization.

- **Variable Depth Search**: A search algorithm that explores more deeply into chosen paths in the search tree, rather than exploring alternative paths at the current level.

Each of these tuners can be used as via:

```python
from keras_tuner_extensionpack.benchmark.functions import shifted_ackley
from keras_tuner_extensionpack.differential_evolution import DifferentialEvolution
import keras_tuner


class MyTuner(DifferentialEvolution):
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
```

### Algorithms are _pre_-tested for:

```python
def shifted_ackley(x: np.ndarray, shift: tuple = (1, 0.5)) -> float:
    """Shifted Ackley function.

    Args:
        x (np.ndarray): Input vector.
        shift (np.ndarray): Shift vector.

    Returns:
        float: Output value.
    """
    return ackley(np.array([x[i] - shift[i] for i in range(len(x))]))
```

```python
def sphere(x: np.ndarray) -> float:
    """Sphere function.

    Args:
        x (np.ndarray): Input vector.

    Returns:
        float: Output value.
    """
    return np.sum(x**2)
```

```python
def rosenbrock(x: np.ndarray) -> float:
    """Rosenbrock function.

    Args:
        x (np.ndarray): Input array of shape with larger than 2,
            representing the coordinates.

    Returns:
        float: Output value.
    """
    return np.sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0, axis=0)
```

## Install

```shell
pip install keras-tuner-extensionpack
```

or:

```shell
pip install git+https://github.com/Anselmoo/keras-tuner-extensionpack
```

> [!NOTE]
> Please note that this is a very early draft of `keras-tuner-extensionpack`. This version may contain incomplete features, bugs, or major changes.

## Contributing

Contributions to `keras-tuner-extensionpack` are welcome!

## [License](LICENSE.md)

---

> [!WARNING]
> This project was partially generated with extended support by GitHub Copilot and may not be completely verified. Please use with caution and feel free to report any issues you encounter. Thank you!
