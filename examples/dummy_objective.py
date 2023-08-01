import math
import random
from pathlib import Path
from typing import List, Dict
import numpy as np


class DummyObjective:
    dummy_curves = {}
    dummy_max_budget = 100
    dummy_is_minimize = False

    @staticmethod
    def get_random_curve() -> np.ndarray:
        values = [random.random() for _ in range(3)]
        values[:2] = sorted(values[:2])
        if DummyObjective.dummy_is_minimize:
            y1 = values[1]
            y2 = values[0]
            alphas = values[2] * y2
        else:
            y1 = values[0]
            y2 = values[1]
            alphas = values[2] * (1 - y2) + y2

        max_budget = DummyObjective.dummy_max_budget
        betas = y2 - alphas
        gammas = math.log((y2 - alphas) / (y1 - alphas)) / math.log(1 / max_budget)

        scaled_budgets = np.arange(1, max_budget + 1) / max_budget
        curve = alphas + betas * np.power(scaled_budgets, -1 * gammas)
        return curve

    @staticmethod
    def dummy_objective_function(configuration: Dict, epoch: int, previous_epoch: int, checkpoint_path: Path) -> List:
        checkpoint_path_str = str(checkpoint_path)
        if checkpoint_path_str not in DummyObjective.dummy_curves:
            DummyObjective.dummy_curves[checkpoint_path_str] = DummyObjective.get_random_curve()

        curve = DummyObjective.dummy_curves[checkpoint_path_str]
        curve_performance = curve[previous_epoch: epoch]
        metric = [{'epoch': e, 'metric': p} for e, p in zip(range(previous_epoch + 1, epoch + 1), curve_performance)]
        return metric
