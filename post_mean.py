import torch
from torch import Tensor, sqrt, exp

from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP
from botorch.models.model import Model
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.means import ZeroMean, ConstantMean
from gpytorch.constraints import GreaterThan

dims = 1        ### Number of dimensions
eps = 1e-9      ### Noise variance level
problem_bounds = torch.stack([torch.zeros(dims), torch.ones(dims)])

def test_func(x):
    f = torch.sin(5 * torch.pi * x**2) + 0.5*x
    return -f.sum(axis=1).view(-1,1)

def torch_lhs(n,ndims):
    t = torch.rand((n,ndims), dtype=torch.float64)
    row = torch.randperm(t.shape[0])
    return torch.cat((t[:,0].view(-1,1),t[row,1].view(-1,1)),-1)

#%% Generate GP-obj
ns = 10*dims
data_xs = torch.rand(ns,dims).to(torch.double)
data_ys = test_func(data_xs)
gp_model = SingleTaskGP(data_xs, data_ys,
                        likelihood=GaussianLikelihood(noise_constraint=GreaterThan(1e-8)),
                        mean_module=ZeroMean(),
                        covar_module=RBFKernel(ard_num_dims=dims),
                    )
mll = ExactMarginalLogLikelihood(gp_model.likelihood, gp_model)
fit_gpytorch_model(mll);

#%%
x_test = torch.rand(3,dims).to(torch.double)
# prior_mean = gp_model.mean_module(x_test[0])
with torch.no_grad():
    Gram_mat = gp_model.covar_module(gp_model.train_inputs[0],gp_model.train_inputs[0]) + (
                gp_model.likelihood.noise*torch.eye(ns)
                )
    # Gram_mat = gp_model.posterior(gp_model.train_inputs[0]).mvn.covariance_matrix 
# Y = torch.linalg.solve(Gram_mat,(gp_model.train_targets-prior_mean)).view(-1,1)
Y = torch.linalg.solve(Gram_mat,gp_model.train_targets).view(-1,1)

#posterior mean manually computed
gp_model.covar_module.covar_dist(x_test, gp_model.train_inputs[0])@Y #+ prior_mean
#posterior mean 
gp_model.posterior(x_test).mean
