"""Type definitions for the extension pack."""

from __future__ import annotations

from typing import Dict
from typing import List
from typing import Optional
from typing import TypeVar
from typing import Union

from keras_tuner.src.engine import hyperparameters as hp_module
from keras_tuner.src.engine.trial import TrialStatus


Value = Union[int, float, bool, str]
Hyperparams = Dict[str, Value]
Population = List[Hyperparams]

TrialDict = Dict[str, Union[TrialStatus, Optional[Hyperparams]]]

HParamT = TypeVar("HParamT", bound=Hyperparams)
PopT = TypeVar("PopT", bound=Population)
TrialT = TypeVar("TrialT", bound=TrialDict)

HP = TypeVar("HP", bound=hp_module.HyperParameters)
