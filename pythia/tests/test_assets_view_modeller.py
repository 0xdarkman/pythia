import numpy as np
import pytest

from pythia.core.visualization.assets_view_modeller import AssetViewModeller


def test_empty_data_range():
    assert len(AssetViewModeller([], [])) == 0


def test_data_mismatch_results_in_error():
    with pytest.raises(AssetViewModeller.MismatchError):
        AssetViewModeller([[1, 2]], [[1, 0]])
    with pytest.raises(AssetViewModeller.MismatchError):
        AssetViewModeller([[1, 2], [2, 3]], [[1, 0], [1, 0]])


def test_length_reflects_number_of_symbols():
    assert len(AssetViewModeller([[1]], [[0, 1]])) == 2


def test_provides_normalized_asset_distribution_including_cash():
    m = AssetViewModeller([[2, 3, 4], [3, 4, 5]], [[3, 1, 0], [0, 2, 2], [5, 3, 2]])
    prices, weights = next(m)
    assert (prices == np.array([1, 1, 1])).all()
    assert (weights == np.array([0.75, 0.0, 0.5])).all()
    prices, weights = next(m)
    assert (prices == np.array([2, 3, 4])).all()
    assert (weights == np.array([0.25, 0.5, 0.3])).all()
    prices, weights = next(m)
    assert (prices == np.array([3, 4, 5])).all().all()
    assert (weights == np.array([0.0, 0.5, 0.2])).all()


def test_use_zeros_for_cash_if_configured():
    m = AssetViewModeller([[1, 2]], [[1, 0], [0, 1]], use_zero_cash=True)
    p, w = next(m)
    assert (p == np.array([0, 0])).all()


def test_relative_weights():
    m = AssetViewModeller([[1, 2]], [[1, 0], [0.5, 0.5]], relative_weights=True)
    p, w = next(m)
    assert (p == np.array([1, 2])).all()
    assert (w == np.array([0.0, 1.0])).all()


def test_iterates_over_all_symbols():
    m = AssetViewModeller([[2, 3], [3, 4]], [[3, 1, 0], [0, 2, 2]])
    count = 0
    for _ in m:
        count += 1

    assert count == len(m)
