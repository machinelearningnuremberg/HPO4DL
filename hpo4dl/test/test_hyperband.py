import pytest
from hpo4dl.optimizers.hyperband.hyperband import HyperBand
from unittest.mock import Mock
import pandas as pd


class TestHyperBand:
    @pytest.fixture
    def configuration_manager(self):
        return Mock()

    @pytest.fixture(params=[True, False])
    def hyperband(self, configuration_manager, request):
        minimization = request.param
        return HyperBand(
            max_budget=81,
            configuration_manager=configuration_manager,
            eta=3,
            seed=42,
            minimization=minimization,
            device="cpu"
        )

    def test_init(self, hyperband):
        assert hyperband.max_budget == 81
        assert hyperband.eta == 3
        assert hyperband.seed == 42
        assert hyperband.device == "cpu"
        assert hyperband.successive_halving_n == [
            [81, 27, 9, 3, 1],
            [34, 11, 3, 1],
            [15, 5, 1],
            [8, 2],
            [5]
        ]
        assert hyperband.successive_halving_r == [
            [1, 3, 9, 27, 81],
            [3, 9, 27, 81],
            [9, 27, 81],
            [27, 81],
            [81]
        ]
        assert hyperband.successive_halving_k == [
            [27, 9, 3, 1, 0],
            [11, 3, 1, 0],
            [5, 1, 0],
            [2, 0],
            [1]
        ]

    def test_get_top_k_configuration_id(self, hyperband):
        data = pd.DataFrame({"config_id": [1, 2, 3, 4], "performance": [0.2, 0.4, 0.1, 0.3]})
        if hyperband.minimization:
            assert hyperband.get_top_k_configuration_id(data, 2) == [3, 1]
        else:
            assert hyperband.get_top_k_configuration_id(data, 2) == [2, 4]

    def test_observe(self, hyperband):
        configuration_id = [1, 2, 3]
        fidelity = [10, 20, 30]
        metric = [{"metric": [0.1, 0.2, 0.3]}] * len(fidelity)
        hyperband.observe(configuration_id, fidelity, metric)
        assert len(hyperband.history) == 9
        assert len(hyperband.bracket_history) == 9

    def test_get_best_configuration_id(self, hyperband):
        hyperband.history = pd.DataFrame({"config_id": [1, 2, 3, 4], "performance": [0.2, 0.4, 0.1, 0.3]})
        if hyperband.minimization:
            assert hyperband.get_best_configuration_id() == 3
        else:
            assert hyperband.get_best_configuration_id() == 2
