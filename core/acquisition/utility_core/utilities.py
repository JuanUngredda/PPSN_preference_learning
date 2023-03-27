
from scipy.stats import dirichlet
import numpy as np


class true_utility_func:
    def __init__(self, n_params, u_func):
        self.n_params = n_params
        self.u_func = u_func

    def __call__(self, y, parameter):
        utility = self.u_func(y, parameter)
        return utility

class Linear_utility_func:
    def __init__(self, n_params):
        self.n_params = n_params

    def __call__(self, y, parameter, vectorized=False):
        # ( Ntheta , Ny , Nm)
        if not vectorized:
            parameter = np.atleast_2d(parameter)[:, np.newaxis, :]
            y = np.atleast_2d(y)[np.newaxis, :, :]
        utility = np.sum(y * parameter, axis=-1)
        return utility



class Tchevichev_utility_func:

    def __init__(self, n_params):
        self.n_params = n_params

    def __call__(self, y, parameter, vectorized=False):
        """
        y: output vector. dimensions (1,Nz,Nx,Ny)
        parameter: parameter vector. dimensions (Ntheta, 1,1,Ny)
        utility: utility vector. dimensions (Ntheta, Nz, Nx)
        """
        if not vectorized:
            parameter = np.atleast_2d(parameter)[:, np.newaxis, :]
            y = np.atleast_2d(y)[np.newaxis, :, :]
        scaled_vectors = parameter * y
        utility = np.min(scaled_vectors, axis=-1)

        return utility


class composed_utility_functions:

    def __init__(self, u_funcs):
        self.u_funcs = u_funcs

    def __call__(self, y, weights, parameters, vectorised=False):
        y = np.atleast_2d(y)
        weights = np.atleast_2d(weights)
        parameters = np.atleast_2d(parameters)

        if vectorised:
            out = 0
            for ufun in range(len(self.u_funcs)):
                objective_weights = weights[:,ufun][:,np.newaxis, np.newaxis]
                individual_utility = self.u_funcs[ufun](y, parameters[ufun][:,np.newaxis,np.newaxis,:],
                                                        vectorized=vectorised)

                out += individual_utility *objective_weights
            return out
        else:

            util = np.zeros((weights.shape[0], y.shape[0], len(self.u_funcs)))

            for ufun in range(len(self.u_funcs)):
                out = self.u_funcs[ufun](y, parameters[ufun])

                util[:, :, ufun] = out
            total_utility = np.sum(util * weights[:, np.newaxis, :], axis=-1)
            return total_utility