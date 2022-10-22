import numpy as np

from dataclasses import dataclass
from typing import Optional


@dataclass
class Perceptron:
    """
        A single neuron with the sigmoid activation function.
        Fields:
            inputs: The number of inputs in a perceptron, not counting the bias.
            weights: Weights for each input values as well as the bias.
            bias: The bias term. 1.0 by default.
    """
    
    inputs: int
    weights: Optional[np.ndarray] = None
    bias: int = 1.0

    def run(self, x):
        """Run the perceptron. x is a Python list with the input values."""
        sum = np.dot(np.append(x, self.bias), self.weights)
        return self.sigmoid(sum)
    
    def set_weights(self, w_init):
        """Set the weights. w_init is a Python list with the weights."""
        self.weights = np.array(w_init)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def __post__init__(self):
        self.weights: Optional[np.ndarray] = (np.random.rand(self.inputs + 1) 
                                              * 2) - 1