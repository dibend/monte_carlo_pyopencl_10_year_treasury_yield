import sys
import types
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Provide a very small stub for the `gradio` package so that importing `app`
# does not require the real dependency. Only the ``Error`` attribute is needed
# for the library code and our tests.
# ---------------------------------------------------------------------------
gr_stub = types.ModuleType("gradio")

class GradioError(Exception):
    """Minimal stand-in for ``gradio.Error``."""

    pass

gr_stub.Error = GradioError
sys.modules.setdefault("gradio", gr_stub)

# Now that the stub is in ``sys.modules`` we can safely import the app module.
import app  # noqa: E402  pylint: disable=wrong-import-position


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def sample_historical_series():
    """Return a tiny synthetic interest-rate series for testing."""
    dates = pd.date_range(start="2024-01-01", periods=10, freq="D")
    values = np.linspace(4.0, 4.5, num=len(dates))  # monotonically increasing
    series = pd.Series(values, index=dates, name=app.SERIES_TO_SIMULATE)
    return series


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_run_monte_carlo_simulation_numpy_shape_and_non_negative(sample_historical_series):
    """The NumPy Monte Carlo engine should return an array with the expected
    shape and with non-negative simulated rates.
    """
    num_simulations = 50
    num_days = 20

    sims = app.run_monte_carlo_simulation_numpy(
        sample_historical_series, num_simulations, num_days
    )

    # Expected shape: (num_days + 1, num_simulations)
    assert sims.shape == (num_days + 1, num_simulations)

    # All simulated interest rates should be non-negative.
    assert np.all(sims >= 0.0)


def test_fetch_data_from_csv_success(monkeypatch):
    """``fetch_data_from_csv`` should return a pandas Series containing the
    requested column and a ``DatetimeIndex`` when the CSV is well-formed.
    """

    # Create a dummy DataFrame resembling the structure of the Treasury CSV
    dates = pd.date_range(start="2024-01-01", periods=3, freq="D")
    df = pd.DataFrame(
        {
            "Date": dates,
            app.SERIES_TO_SIMULATE: [4.1, 4.15, 4.2],
        }
    )

    # Monkeypatch ``pandas.read_csv`` so the function uses our dummy data.
    monkeypatch.setattr(pd, "read_csv", lambda _: df.copy())

    series = app.fetch_data_from_csv("dummy://url")

    # Basic sanity checks on the returned data structure.
    assert isinstance(series, pd.Series)
    assert not series.empty
    assert series.name == app.SERIES_TO_SIMULATE
    assert isinstance(series.index[0], pd.Timestamp)


def test_get_opencl_devices_contains_cpu():
    """The helper should always return at least the CPU option."""
    devices = app.get_opencl_devices()
    assert "CPU (Numpy)" in devices