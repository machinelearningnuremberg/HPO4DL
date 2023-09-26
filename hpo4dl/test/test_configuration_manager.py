import pytest
import ConfigSpace as CS
from unittest.mock import Mock
from hpo4dl.configuration_manager.configuration_manager import ConfigurationManager


# Assuming your_module is the module where ConfigurationManager is

class TestConfigurationManager:
    @pytest.fixture
    def seed(self):
        return 42

    @pytest.fixture(params=[1, 2])
    def config_space_index(self, request):
        return request.param

    @pytest.fixture
    def config_space(self, seed, config_space_index):
        config_space = CS.ConfigurationSpace(seed=seed)
        if config_space_index == 1:
            config_space.add_hyperparameters([
                CS.UniformFloatHyperparameter('lr', lower=1e-4, upper=0.1, log=True),
                CS.UniformFloatHyperparameter('weight_decay', lower=1e-5, upper=0.1, log=False),
                CS.UniformFloatHyperparameter('momentum', lower=0.1, upper=0.99, log=True),
            ])
        elif config_space_index == 2:
            config_space.add_hyperparameters([
                CS.UniformIntegerHyperparameter('width', lower=4, upper=10, log=False),
            ])
        elif config_space_index == 3:
            config_space.add_hyperparameters([
                CS.UniformIntegerHyperparameter('width', lower=2, upper=4, log=False),
            ])
        else:
            raise NotImplementedError
        return config_space

    @pytest.fixture
    def configuration_manager(self, config_space, seed, config_space_index):
        if config_space_index != 3:
            return ConfigurationManager(
                configuration_space=config_space,
                num_configurations=5,
                seed=seed
            )

    def test_configuration_manager_error(self, config_space, seed, config_space_index):
        if config_space_index == 3:
            with pytest.raises(
                RuntimeError,
                match=f'Unable to add 5 unique configurations, '
                      f'only 3 could be added due to duplicates.'
            ):
                manager = ConfigurationManager(
                    configuration_space=config_space,
                    num_configurations=5,
                    seed=seed
                )

    def test_init(self, configuration_manager, config_space, config_space_index):
        if config_space_index == 1:
            expected_value = 5
        elif config_space_index == 2:
            expected_value = 5
        elif config_space_index == 3:
            return
        else:
            raise NotImplementedError

        assert configuration_manager.configuration_space == config_space
        assert configuration_manager.num_configurations == 5
        assert configuration_manager.seed == 42
        assert len(configuration_manager.configurations) == expected_value
        assert len(configuration_manager.configurations_set) == expected_value
        assert len(configuration_manager.configurations_df) == expected_value
        self.assert_configurations_are_unique(configuration_manager.configurations)

    def test_generate_configurations(self, configuration_manager):
        configs = configuration_manager.generate_configurations(5)
        assert len(configs) == 5

    def test_get_configurations(self, configuration_manager, config_space_index):
        if config_space_index == 1:
            expected_value = 5
        elif config_space_index == 2:
            expected_value = 5
        else:
            raise NotImplementedError

        configs_df = configuration_manager.get_configurations()
        assert not configs_df.empty
        assert len(configs_df) == expected_value

    def test_get_configuration(self, configuration_manager):
        config = configuration_manager.get_configuration(0)
        assert config is not None

    def test_add_configurations(self, configuration_manager, config_space_index):
        if config_space_index == 1:
            expected_value = 10
        elif config_space_index == 2:
            expected_value = 7
        else:
            raise NotImplementedError

        if config_space_index == 2:
            with pytest.raises(
                RuntimeError,
                match=f'Unable to add 5 unique configurations, '
                      f'only 2 could be added due to duplicates.'
            ):
                configuration_manager.add_configurations(5)
        else:
            configuration_manager.add_configurations(5)

        assert len(configuration_manager.configurations) == expected_value
        assert len(configuration_manager.configurations_set) == expected_value
        assert len(configuration_manager.get_configurations()) == expected_value
        self.assert_configurations_are_unique(configuration_manager.configurations)

    @staticmethod
    def assert_configurations_are_unique(configurations):
        configs_as_tuples = [tuple(config.items()) for config in configurations]
        unique_configs = set(configs_as_tuples)
        assert len(configurations) == len(unique_configs), "Some configurations are duplicated"
