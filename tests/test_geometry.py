import numpy as np
import pytest

from molecular_simulation_tools.geometry import discretize_cell_length

discretization_data = [
    (1.0, 2, np.array([0.25, 0.75])),
    (1.0, 3, np.array([1.0 / 6.0, 0.5, 5.0 / 6.0])),
    (1.0, 4, np.array([1.0 / 8.0, 3.0 / 8.0, 5.0 / 8.0, 7.0 / 8.0])),
]


@pytest.mark.parametrize("length, num, expected_output", discretization_data)
def test_discretize_cell_length(length, num, expected_output):
    output = discretize_cell_length(length, num)

    if num % 2 == 1:
        assert output[num // 2] == length / 2

    # Make sure that the spacing is as far to the right as it is to the left.
    assert np.allclose(length - output[-1], output[0])
    assert np.allclose(output, expected_output)
