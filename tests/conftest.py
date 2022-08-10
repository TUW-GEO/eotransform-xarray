import gc
import random
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.append((Path(__file__).parent / "helpers").as_posix())


@pytest.fixture
def disabled_gc():
    try:
        gc.disable()
        yield
    finally:
        gc.enable()


@pytest.fixture(scope='module')
def fixed_seed():
    py_state = random.getstate()
    np_state = np.random.get_state()
    random.seed(42)
    np.random.seed(42)
    yield 42
    random.setstate(py_state)
    np.random.set_state(np_state)
