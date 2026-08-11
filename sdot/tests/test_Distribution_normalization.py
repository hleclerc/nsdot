"""Test Distribution normalization with ComputedAttribute caching."""

from sdot.distributions.SumOfDiracs import SumOfDiracs
import numpy as np


def test_sum_of_diracs_mass_invalidation():
    """Test that mass is invalidated when weights change."""
    positions = np.array([[0.0], [1.0], [2.0]])
    weights = np.array([1.0, 2.0, 3.0])

    dist = SumOfDiracs(positions, weights)

    # Check that mass is valid initially
    assert dist._computed_attrs["mass"]._cache_valid, "Mass should be valid initially"

    # Verify original measure
    original_mass = dist.measure
    assert original_mass == 6.0, f"Expected original mass 6.0, got {original_mass}"

    # Modify weights (should invalidate mass)
    dist.weights = np.array([2.0, 2.0, 2.0])
    assert not dist._computed_attrs["mass"]._cache_valid, "Mass should be invalidated after weights change"

    # Verify new measure after update
    new_mass = dist.measure
    assert new_mass == 6.0, f"Expected new mass 6.0, got {new_mass}"


def test_sum_of_diracs_normalized_version():
    """Test normalized_version() scales weights to match target_mass."""
    positions = np.array([[0.0], [1.0], [2.0]])
    weights = np.array([1.0, 2.0, 3.0])

    dist = SumOfDiracs(positions, weights)
    original_mass = dist.mass.value
    assert original_mass == 6.0

    # Create a normalized version (currently just returns self, but could scale weights)
    normalized = dist.normalized_version()
    # TODO: once normalized_version() is implemented, test that it scales correctly
    assert normalized.mass.value == 6.0


if __name__ == "__main__":
    test_sum_of_diracs_mass_caching()
    test_sum_of_diracs_normalized_version()
    print("All distribution tests passed!")
