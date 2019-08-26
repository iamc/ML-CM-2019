import numpy as np
import torch

from .template import Target


class Wave(Target):
    def __init__(self):
        super().__init__(2, "Wave")

    def energy(self, x):
        w = torch.sin(np.pi * x[:, 0] / 2.0)
        return -0.5 * ((x[:, 1] - w) / 0.4) ** 2
