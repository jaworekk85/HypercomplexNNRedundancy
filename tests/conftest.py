from scripts.runtime import configure

import pytest


@pytest.fixture(autouse=True)
def deterministic_seed():
    configure(2026)
