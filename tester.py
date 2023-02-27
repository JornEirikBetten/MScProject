import numpy as np
from numpy.random import default_rng


rng = default_rng()
N = 5
indices = np.arange(N)
rng.shuffle(indices)
batch_indices = np.split(indices, 5)
print(batch_indices)
