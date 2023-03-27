import pandas as pd
import os
from pyDOE import *


class AcquisitionFunctionandDecisionMakerInteraction():
    def __init__(self, model, true_f, acquisition_optimiser, acquisition_f, space, seed=None, InteractionClass=None,
                 Inference_Object=None, path=None):

        self.model = model
        self.objective = true_f
        self.acq_opt = acquisition_optimiser
        self.n_f = model.output_dim
        self.seed = seed
        self.acquisition_f = acquisition_f
        self.data = {}
        self.data["Utility"] = np.array([])
        self.data["Utility_sampled"] = np.array([])
        self.data["Best_Utility"] = np.array([])
        self.space = space
        self.InteractionClass = InteractionClass
        self.Inference_Object = Inference_Object
        self.path = path

    def get_optimiser(self):
        return self.acq_opt.optimize_inner_func

    def get_true_parameters(self):
        return self.InteractionClass.get_true_parameters()

    def get_algorithm_utility(self):
        return self.Inference_Object.get_utility_function()

    def get_true_utility_function(self):
        return self.InteractionClass.get_true_utility_function()

    def include_posterior_samples_in_acq(self, posterior_samples):
        self.acquisition_f.include_fantasised_posterior_samples(posterior_samples)

    def get_data(self):
        return self.InteractionClass.get_

    def _update_model(self, X, Y, model=None):
        """
        Updates the model (when more than one observation is available) and saves the parameters (if available).
        """

        ### --- input that goes into the model (is unziped in case there are categorical variables)
        X_inmodel = self.space.unzip_inputs(X)
        model.updateModel(X_inmodel, Y)

    def get_DecisionMaker_data(self):
        return self.InteractionClass.ShowParetoFronttotheDecisionMaker()

    def posterior_sample_generation(self, solution_picked_dm_index, PF):

        self.Inference_Object.update_sampled_data(PF, solution_picked_dm_index)
        posterior_samples = self.Inference_Object.posterior_sampler(n_samples=50)
        return posterior_samples

    def acquisition_function(self, X):
        return -self.acquisition_f._compute_acq(X)

    def store_results(self, recommended_x):

        true_underlying_utility = self.get_true_utility()
        true_parameters = self.get_true_parameters()

        Y_recommended, cost_new = self.objective.evaluate(recommended_x)
        Y_recommended = np.concatenate(Y_recommended, axis=1)
        true_recommended_utility = true_underlying_utility(y=Y_recommended,
                                                           weights=true_parameters[1],
                                                           parameters=true_parameters[0])
        out = true_recommended_utility.reshape(-1)

        self.data["Utility"] = np.concatenate((self.data["Utility"], np.array(out).reshape(-1)))

        Y_train = self.model.get_Y_values()
        Y_train = np.concatenate(Y_train, axis=1)
        uval_sampled = np.max(true_underlying_utility(y=Y_train,
                                                      weights=true_parameters[1],
                                                      parameters=true_parameters[0]))

        N_entries = len(self.data["Utility"].reshape(-1))
        self.data["Utility_sampled"] = np.concatenate(
            (self.data["Utility_sampled"], np.array(uval_sampled).reshape(-1)))

        true_best_x, true_best_val = self.compute_underlying_best()
        self.data["Best_Utility"] = np.repeat(true_best_val, N_entries)

        if self.path is not None:
            gen_file = pd.DataFrame.from_dict(self.data)
            results_folder = "Utility"

            path = self.path + "/" + results_folder + '/it_' + str(self.seed) + '.csv'
            if os.path.isdir(self.path + "/" + results_folder) == False:
                os.makedirs(self.path + "/" + results_folder)

            gen_file.to_csv(path_or_buf=path)

            extra_data = {"parameters": true_parameters, "numberofsteps": self.NLastSteps}
            extra_gen_file = pd.DataFrame.from_dict(extra_data)
            extra_path = self.path + "/" + results_folder + '/parameters_' + str(self.seed) + '.csv'
            extra_gen_file.to_csv(path_or_buf=extra_path)

    def compute_underlying_best(self):

        true_underlying_utility = self.get_true_utility()
        weight = self.get_true_parameters()

        def top_true_utility(X):
            X = np.atleast_2d(X)
            Y_recommended, cost_new = self.objective.evaluate(X)
            Y_recommended = -np.concatenate(Y_recommended, axis=1)
            uval = true_underlying_utility(y=Y_recommended,
                                           weights=weight[1],
                                           parameters=weight[0])

            return np.array(uval).reshape(-1)

        true_best_x, true_best_val = self.acq_opt.optimize_inner_func(f=top_true_utility)

        return true_best_x, true_best_val
