import torch
import numpy as np

# This model trains the discriminator on a 'real' dataset of a bell curve and initially provides the generator with uniform data
# Both the discriminator and the generator use a sigmoid activation function and train a linear transformation

# Define functions that generate a 'real' and 'random' set of datapoints
mu = 5
sigma = 2
real = lambda n: torch.Tensor(np.random.normal(mu, sigma, (n, 1)))
print(type(real))

# Define the discriminator


# Define the generator


# Train both models