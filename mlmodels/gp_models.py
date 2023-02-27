import math
import torch
import gpytorch
from matplotlib import pyplot as plt
torch.set_default_dtype(torch.float64)


"""
-------------------------------------------------------------------------------
                    Exact Gaussian Process Model
-------------------------------------------------------------------------------
"""

class ExactGP(gpytorch.models.ExactGP):
    def __init__(self, xtrain, ytrain, likelihood, mean=gpytorch.means.ConstantMean(), ard_dims=1):
        super(ExactGP, self).__init__(xtrain, ytrain, likelihood)
        self.mean_module = mean
        self.covar_module = gpytorch.kernels.RBFKernel(ard_dims=ard_dims)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


"""
--------------------------------------------------------------------------------
        Neural Network kernel with Exact GP Inference as final layer
--------------------------------------------------------------------------------
"""

"""
                Neural Network kernel
"""
class FeatureExtractor(torch.nn.Sequential):
    def __init__(self, nfeatures, nfeatures_out):
        super(FeatureExtractor, self).__init__()
        self.add_module('linear1', torch.nn.Linear(nfeatures, 100))
        self.add_module('relu1', torch.nn.ReLU())
        self.add_module('linear2', torch.nn.Linear(100, 50))
        self.add_module('relu2', torch.nn.ReLU())
        self.add_module('linear3', torch.nn.Linear(50, n_features_out))


class NNKernel(gpytorch.models.ExactGP):
    def __init__(self,
                 xtrain,
                 ytrain,
                 likelihood,
                 mean=gpytorch.means.ConstantMean(),
                 last_layer_dim = 3
                 ):
        super(NNKernel, self).__init__(xtrain, ytrain, likelihood)
        self.mean_module = mean
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.GridInterpolationKernel(
                gpytorch.kernels.RBFKernel(ard_dims=last_layer_dim),
                num_dims=last_layer_dim,
                grid_size=100
                )
        )
        nfeatures = xtrain.size(-1)
        self.feature_extractor = FeatureExtractor(nfeatures, last_layer_dim)
        # Scaling of features output from NN
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1., 1.)

    def forward(self, x):
        # Through the features extractor
        projected_x = self.feature_extractor(x)
        projected_x = self.scale_to_bounds(projected_x)

        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
