from pygmo import *
from pyDOE import *
import numpy as np
import matplotlib.pyplot as plt
#DM utilities. These functions are not vectorised.
from .utilities import composed_utility_functions


class ParetoFrontGeneration():

    def __init__(self, model, space, seed, utility, show_optimistic_front=False):

        #initialising and passing variables
        self.output_dimensions = model.output_dim

        #passing classes and functions
        self.model = model
        self.space = space
        self.DM_utility = composed_utility_functions(utility)

        #initialising values
        self.show_optimistic_front = show_optimistic_front
        self.wdim = len(utility)
        self.tindvdim = [u.n_params for u in utility]
        self.tdim = np.sum([u.n_params for u in utility])
        self.m_dim = self.tdim + self.wdim
        self.weight = self.prior_sampler(n_samples=1, seed=seed)#([np.array([[0., 1]])], np.array([[1.]]))#
        print("weight", self.weight)

    def get_true_parameters(self):
        return self.weight

    def get_true_utility_values(self):
        weight = self.get_true_parameters()
        def utility(y):
            return self.DM_utility(y, weights=self.weight[1],
                                      parameters=self.weight[0])
        return utility

    def get_true_utility_function(self):
        return self.DM_utility

    def dirich_sampler(self, dim, n_samples=1, seed=None):

        if seed is None:
            samples = np.random.dirichlet(np.ones((dim,)), n_samples)
        else:
            random_state = np.random.RandomState(seed)
            samples = random_state.dirichlet(np.ones((dim,)), n_samples)
        return samples

    def prior_sampler(self, n_samples, seed=None):
        if seed is None:
            theta_samples = [self.dirich_sampler(dim=d, n_samples=n_samples) for d in self.tindvdim]
            weight_samples = self.dirich_sampler(self.wdim, n_samples=n_samples)
        else:

            theta_samples = [self.dirich_sampler(d, n_samples=n_samples, seed=seed) for d in self.tindvdim]
            weight_samples = self.dirich_sampler(self.wdim, n_samples=n_samples, seed=seed)

        return theta_samples, weight_samples


    def Generate_Pareto_Front(self):
        # X_train = self.model.get_X_values()

        if self.show_optimistic_front:
            GP_y_predictions = self.quantile_prediction_model(self.model)
        else:
            GP_y_predictions  = self.mean_prediction_model(self.model)


        bounds = self.space.get_continuous_bounds()
        bounds = self.bounds_format_adapter(bounds)
        udp = GA(f=GP_y_predictions, bounds=bounds, n_obj=self.output_dimensions)
        pop = population(prob=udp, size=100)#100
        algo = algorithm(nsga2(gen=300))#300
        pop = algo.evolve(pop)
        fits, vectors = pop.get_f(), pop.get_x()
        ndf, dl, dc, ndr = fast_non_dominated_sorting(fits)
        result_x = vectors[ndf[0]]
        result_fx = fits[ndf[0]]

        return -result_fx, result_x  # GP_y_predictions

    def bounds_format_adapter(self, bounds):
        bounds = np.array(bounds)
        bounds_correct_format = []
        for b in range(bounds.shape[0]):
            bounds_correct_format.append(list(bounds[:, b]))
        return bounds_correct_format

    def mean_prediction_model(self, model):
        def prediction(X):
            X = np.atleast_2d(X)

            mu_x = model.posterior_mean(X)
            mu_x = np.vstack(mu_x).T
            return mu_x
        return prediction

    def quantile_prediction_model(self, model):
        def prediction(X):
            X = np.atleast_2d(X)
            mu_x = model.posterior_mean(X)
            mu_x = np.vstack(mu_x).T

            var_x = model.posterior_variance(X, noise=False)
            var_x = np.vstack(var_x).T
            sqrt = np.sqrt(var_x)

            return mu_x + 1.96 * sqrt
        return prediction

    def ShowParetoFronttotheDecisionMaker(self):

        PF, PF_Xvals = self.Generate_Pareto_Front()  # generate pareto front
        Utility_PF =  self.DM_utility(PF,
                                      weights=self.weight[1],
                                      parameters=self.weight[0])  # dm picks a solution

        # solution picked by the dm in output space
        solution_picked_dm_index = np.argmax(Utility_PF)  # minimise utility
        solution_x_picked_dm = PF_Xvals[solution_picked_dm_index]

        return solution_picked_dm_index, solution_x_picked_dm, PF, Utility_PF

class GA:
    # Define objectives
    def __init__(self, f, bounds, n_obj):
        self.f = f
        self.bounds = bounds
        self.n_obj = n_obj
    def fitness(self, x):
        x = np.atleast_2d(x)
        output = self.f(x)
        vect_func_val = []
        for i in range(self.n_obj):
            vect_func_val.append(output[:,i])
        return -np.array(vect_func_val).reshape(-1)

    # Return number of objectives
    def get_nobj(self):
        return self.n_obj

    # Return bounds of decision variables
    def get_bounds(self):
        return self.bounds  # ([0]*1, [2]*1)

    # Return function name
    def get_name(self):
        return "INNER OPTIMISATION PROBLEM"
