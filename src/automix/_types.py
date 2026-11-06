"""Shared array types.

Audio is ``float32`` shaped ``(channels, samples)``, matching pedalboard's convention.
"""

import numpy as np
import numpy.typing as npt

Audio = npt.NDArray[np.float32]
SampleIndices = npt.NDArray[np.int64]
