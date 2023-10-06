# Hyperparameter Optimization for Deep Learning (HPO4DL)

HPO4DL is a framework for multi-fidelity (gray-box) hyperparameter optimization.
The core optimizer in HPO4DL is DyHPO, a novel Bayesian Optimization approach to
hyperparameter optimization tailored for deep learning. DyHPO dynamically determines
the best hyperparameter configurations to train further by using a deep kernel for
Gaussian Processes that captures the details of the learning curve and an acquisition function
that incorporates multi-budget information.

## Installation

To install the package:

```bash
pip install hpo4dl
```

## Getting started

Following is a simple example to get you started:

```python
from typing import List, Dict, Union
from hpo4dl.tuner import Tuner
from ConfigSpace import ConfigurationSpace


def objective_function(
    configuration: Dict,
    epoch: int,
    previous_epoch: int,
    checkpoint_path: str
) -> Union[Dict, List[Dict]]:
    x = configuration["x"]
    evaluated_info = [
        {'epoch': i, 'metric': (x - 2) ** 2}
        for i in range(previous_epoch + 1, epoch + 1)
    ]
    return evaluated_info


configspace = ConfigurationSpace({"x": (-5.0, 10.0)})

tuner = Tuner(
    objective_function=objective_function,
    configuration_space=configspace,
    minimize=True,
    max_budget=1000,
    optimizer='dyhpo',
    seed=0,
    max_epochs=27,
    num_configurations=1000,
    output_path='hpo4dl_results',
)

incumbent = tuner.run()

```

Key Parameters Explained:

- ```objective_function```: The function you aim to optimize.

- ```configuration_space```: The hyperparameter configuration space over which the optimization is performed.

- ```minimize```: Boolean flag indicates whether the objective function should be minimized (True) or maximized (False).

- ```max_budget```: The cumulative number of epochs the tuner will evaluate. This budget gets distributed across various
  hyperparameter configurations.

- ```optimizer```: Specifies the optimization technique employed.

- ```seed```: Random seed for reproducibility.

- ```max_epochs```: Maximum number of epochs a single configuration is evaluated.

- ```num_configurations```: Determines the number of configurations DyHPO reviews before selecting the next one for
  evaluation. Essentially, it guides the balance between exploration and exploitation in the optimization process.

- ```output_path```: Designates the location to save the results and the checkpoint for the best hyperparameter
  optimization.

### Objective function

```python
def objective_function(
    configuration: Dict,
    epoch: int,
    previous_epoch: int,
    checkpoint_path: str
) -> Union[Dict, List[Dict]]
```

The objective function is tailored to support interrupted and resumed training processes.
Specifically, it should continue training from a ```previous_epoch``` to the designated ```epoch```.

The function should return a dictionary or a list of dictionaries upon completion.
Every dictionary must include the ```epoch``` and ```metric``` keys. Here's a sample return value:

```
{
    “epoch”: 5,
    “metric”: 0.76
}
```

For optimal performance with DyHPO, ensure the metric is normalized.

Lastly, the ```checkpoint_path``` is allocated for saving any intermediate files produced
during training pertinent to the current configuration. It facilitates storing models, logs,
and other relevant data, ensuring that training can resume seamlessly.

### Detailed Examples

For a detailed exploration of the HPO4DL framework, we've provided an in-depth example
under: ```examples/timm_main.py```

To execute the provided example, use the following command:

```bash
python examples/timm_main.py 
    --dataset torch/cifar100 
    --train-split train 
    --val-split validation 
    --optimizer dyhpo  
    --output-dir ./hpo4dl_results

```

## Citation

```
@inproceedings{
wistuba2022supervising,
title={Supervising the Multi-Fidelity Race of Hyperparameter Configurations},
author={Martin Wistuba and Arlind Kadra and Josif Grabocka},
booktitle={Thirty-Sixth Conference on Neural Information Processing Systems},
year={2022},
url={https://openreview.net/forum?id=0Fe7bAWmJr}
}
```