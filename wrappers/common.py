from typing import Tuple, Union

import numpy as np

# Gymnasium step: (obs, reward, terminated, truncated, info)
# Legacy gym step: (obs, reward, done, info)
TimeStep = Union[
    Tuple[np.ndarray, float, bool, bool, dict],
    Tuple[np.ndarray, float, bool, dict],
]
