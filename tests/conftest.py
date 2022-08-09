import gc
import sys
from pathlib import Path

import pytest

sys.path.append((Path(__file__).parent / "helpers").as_posix())


@pytest.fixture
def disabled_gc():
    try:
        gc.disable()
        yield
    finally:
        gc.enable()
