import numpy as np
import time
from GPyOpt.experiment_design import initial_design
from GPyOpt.core.errors import InvalidConfigError
from GPyOpt.optimization.acquisition_optimizer import ContextManager
from pygmo import hypervolume
import pandas as pd
import os

try:
    from GPyOpt.plotting.plots_bo import plot_acquisition, plot_convergence
except:
    pass


class BO(object):
    """
    Runner of the multi-attribute Bayesian optimization loop. This class wraps the optimization loop around the different handlers.
    :param model: GPyOpt model class.
    :param space: GPyOpt space class.
    :param objective: GPyOpt objective class.
    :param acquisition: GPyOpt acquisition class.
    :param evaluator: GPyOpt evaluator class.
    :param X_init: 2d numpy array containing the initial inputs (one per row) of the model.
    :param Y_init: 2d numpy array containing the initial outputs (one per row) of the model.
    :param cost: GPyOpt cost class (default, none).
    :param normalize_Y: whether to normalize the outputs before performing any optimization (default, True).
    :param model_update_interval: interval of collected observations after which the model is updated (default, 1).
    :param de_duplication: GPyOpt DuplicateManager class. Avoids re-evaluating the objective at previous, pending or infeasible locations (default, False).
    """

    def __init__(self, model,
                 space,
                 objective,
                 evaluator,
                 acquisition,
                 X_init,
                 expensive=False,
                 Y_init=None,
                 model_update_interval=1,
                 deterministic=True,
                 DecisionMakerInteractor=None):

        self.acquisition = acquisition
        self.model = model
        self.space = space
        self.objective = objective
        self.evaluator = evaluator
        self.model_update_interval = model_update_interval
        self.X = X_init
        self.Y = Y_init
        self.deterministic = deterministic
        self.model_parameters_iterations = None
        self.expensive = expensive
        self.DecisionMakerInteractor = DecisionMakerInteractor

        print("name of acquisition function wasnt provided")
        self.sample_from_acq = False
        self.tag_last_evaluation = False

    def suggest_next_locations(self, context=None, pending_X=None, ignored_X=None):
        """
        Run a single optimization step and return the next locations to evaluate the objective.
        Number of suggested locations equals to batch_size.

        :param context: fixes specified variables to a particular context (values) for the optimization run (default, None).
        :param pending_X: matrix of input configurations that are in a pending state (i.e., do not have an evaluation yet) (default, None).
        :param ignored_X: matrix of input configurations that the user black-lists, i.e., those configurations will not be suggested again (default, None).
        """
        self.model_parameters_iterations = None
        self.num_acquisitions = 0
        self.context = context
        self._update_model(self.normalization_type)

        suggested_locations = self._compute_next_evaluations(pending_zipped_X=pending_X, ignored_zipped_X=ignored_X)
        return suggested_locations

    def run_optimization(self, max_iter=1,
                         max_time=np.inf,
                         rep=None,
                         eps=1e-8,
                         context=None,
                         verbosity=False,
                         model_average=False,
                         path=None,
                         evaluations_file=None,
                         max_number_DMqueries=0,
                         first_query_iteration=0):
        """
        Runs Bayesian Optimization for a number 'max_iter' of iterations (after the initial exploration data)

        :param max_iter: exploration horizon, or number of acquisitions. If nothing is provided optimizes the current acquisition.
        :param max_time: maximum exploration horizon in seconds.
        :param eps: minimum distance between two consecutive x's to keep running the model.
        :param context: fixes specified variables to a particular context (values) for the optimization run (default, None).
        :param verbosity: flag to print the optimization results after each iteration (default, False).
        :param evaluations_file: filename of the file where the evaluated points and corresponding evaluations are saved (default, None).
        """

        if self.objective is None:
            raise InvalidConfigError("Cannot run the optimization loop without the objective function")

        # --- Save the options to print and save the results
        self.verbosity = verbosity
        self.evaluations_file = evaluations_file
        self.context = context
        self.path = path
        self.rep = rep
        # --- Setting up stop conditions
        self.eps = eps
        self.model_average = model_average
        if (max_iter is None) and (max_time is None):
            self.max_iter = 0
            self.max_time = np.inf
        elif (max_iter is None) and (max_time is not None):
            self.max_iter = np.inf
            self.max_time = max_time
        elif (max_iter is not None) and (max_time is None):
            self.max_iter = max_iter
            self.max_time = np.inf
        else:
            self.max_iter = max_iter
            self.max_time = max_time

        # --- Initial function evaluation and model fitting
        if self.X is not None and self.Y is None:
            self.Y, cost_values = self.objective.evaluate(self.X)

        # --- Initialize iterations and running time
        self.time_zero = time.time()
        self.cum_time = 0
        self.num_acquisitions = 0
        self.suggested_sample = self.X
        self.Y_new = self.Y
        self.Opportunity_Cost = {"Hypervolume": np.array([])}
        self.data = {}

        self.data["Utility_sampled"] = np.array([])
        self.data["Utility"] = np.array([])
        self.data["Best_Utility"] = np.array([])
        self.data["Utility_sampled_all_front"] = np.array([])
        value_so_far = []

        # --- Initialize time cost of the evaluations
        print("MAIN LOOP STARTS")
        self.true_best_stats = {"true_best": [], "mean_gp": [], "std gp": [], "pf": [], "mu_pf": [], "var_pf": [],
                                "residual_noise": []}
        self._update_model()

        # Decision maker's variable initialisation and values
        Pareto_fronts = []
        picked_solution_indeces = []
        self.number_of_queries_taken = 0
        query_schedule = np.arange(first_query_iteration, first_query_iteration + max_number_DMqueries)
        while (self.max_iter > self.num_acquisitions):

            # self.optimize_final_evaluation()
            print("maKG optimizer")
            start = time.time()

            if (self.DecisionMakerInteractor is not None) \
                    and (self.num_acquisitions in query_schedule):
                # get data from decision maker
                solution_picked_dm_index, solution_x_picked_dm, PF, _ = self.DecisionMakerInteractor.get_DecisionMaker_data()

                # include new data
                Pareto_fronts.append(PF)
                picked_solution_indeces.append(solution_picked_dm_index)

                # Inference on pareto front and getting posterior samples
                posterior_samples = self.DecisionMakerInteractor.posterior_sample_generation(picked_solution_indeces,
                                                                                             Pareto_fronts)

                # Include new posterior samples in the acquisition function
                self.DecisionMakerInteractor.include_posterior_samples_in_acq(posterior_samples=posterior_samples)
                self.number_of_queries_taken += 1

            self.suggested_sample = self._compute_next_evaluations()
            print("self.suggested_sample", self.suggested_sample)

            finish = time.time()
            print("time optimisation point X", finish - start)

            self.X = np.vstack((self.X, self.suggested_sample))
            # --- Evaluate *f* in X, augment Y and update cost function (if needed)
            self.evaluate_objective()

            self._update_model()
            # --- Update current evaluation time and function evaluations
            self.cum_time = time.time() - self.time_zero
            self.num_acquisitions += 1
            print("optimize_final_evaluation")

            self.store_results(self.suggested_sample)

            print("self.X, self.Y, self.C , self.Opportunity_Cost", self.X, self.Y, self.Opportunity_Cost)

        return self.X, self.Y, self.Opportunity_Cost

    def store_results(self, recommended_x):

        if self.DecisionMakerInteractor is None:

            P = self.model.get_Y_values()
            P_cur = (-np.concatenate(P, axis=1)).tolist()

            print("P_cur", P_cur)
            print("ref", self.acquisition.ref_point)
            HV0_func = hypervolume(P_cur)
            uval_sampled = HV0_func.compute(ref_point=self.acquisition.ref_point)

        else:
            true_underlying_utility = self.get_true_utility_function()
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

            if self.model_average:
                expected_utility = self.compute_expected_utility_MA()
            else:
                expected_utility = self.compute_expected_utility()

            recommended_Y = Y_train[np.argmax(expected_utility.reshape(-1))]

            uval_sampled = true_underlying_utility(y=recommended_Y,
                                                   weights=true_parameters[1],
                                                   parameters=true_parameters[0])

            uval_sampled_all_front = np.max(true_underlying_utility(y=Y_train,
                                                                    weights=true_parameters[1],
                                                                    parameters=true_parameters[0]))

            N_entries = len(self.data["Utility"].reshape(-1))
            true_best_x, true_best_val = self.compute_underlying_best()
            self.data["Best_Utility"] = np.concatenate((self.data["Best_Utility"], np.array(true_best_val).reshape(-1)))

            self.data["Utility_sampled_all_front"] = np.concatenate(
                (self.data["Utility_sampled_all_front"], np.array(uval_sampled_all_front).reshape(-1)))

        self.data["Utility_sampled"] = np.concatenate(
            (self.data["Utility_sampled"], np.array(uval_sampled).reshape(-1)))

        if self.path is not None:
            gen_file = pd.DataFrame.from_dict(self.data)
            results_folder = "Hypervolume_improve"

            path = self.path + "/" + results_folder + '/it_' + str(self.rep) + '.csv'
            if os.path.isdir(self.path + "/" + results_folder) == False:
                os.makedirs(self.path + "/" + results_folder, exist_ok=True)

            gen_file.to_csv(path_or_buf=path)

            Y_train = self.model.get_Y_values()
            Y_train = np.concatenate(Y_train, axis=1)
            data_path = self.path + "/" + results_folder + '/sampled_Y_' + str(self.rep) + '.csv'
            np.savetxt(data_path, Y_train, delimiter=",")

            best_Y, cost_new = self.objective.evaluate(true_best_x)
            best_Y = np.concatenate(best_Y, axis=1)
            print(best_Y)

            data_path = self.path + "/" + results_folder + '/true_best_Y_' + str(self.rep) + '.csv'
            np.savetxt(data_path, best_Y, delimiter=",")

            data_path = self.path + "/" + results_folder + '/true_best_utility_' + str(self.rep) + '.csv'
            np.savetxt(data_path, true_best_val - uval_sampled_all_front, delimiter=",")
        print("path", self.path + "/" + results_folder)

    def compute_expected_utility(self):
        posterior_samples = self.acquisition.get_posterior_samples()
        utility_parameters = posterior_samples[0]
        linear_weight_combination = posterior_samples[1]
        utility = self.acquisition.Inference_Object.get_utility_function()
        Y_train = self.model.get_Y_values()
        Y_train = np.concatenate(Y_train, axis=1)
        Utility = self.acquisition.utility(y=Y_train,
                                           weights=linear_weight_combination,
                                           parameters=utility_parameters,
                                           vectorised=True)

        expected_utility = np.mean(Utility, axis=0)
        return expected_utility

    def compute_expected_utility_MA(self):
        probability_models, posterior_parameter_samples, models = self.acquisition.Inference_Object.get_utility_information()

        utility_list = []
        for utility_idx, utility in enumerate(models):
            Y_train = self.model.get_Y_values()
            Y_train = np.concatenate(Y_train, axis=1)
            Utility = utility(y=Y_train,
                              weights=np.ones((posterior_parameter_samples[utility_idx][0].shape[0], 1)),
                              parameters=posterior_parameter_samples[utility_idx],
                              vectorised=True)

            expected_utility = np.mean(Utility, axis=0) * probability_models[utility_idx]
            utility_list.append(expected_utility)

        mean_utility = np.sum(utility_list, axis=0)

        return mean_utility

    def get_true_utility_function(self):
        return self.DecisionMakerInteractor.get_true_utility_function()

    def get_true_parameters(self):
        return self.DecisionMakerInteractor.get_true_parameters()

    def get_optimiser(self):
        return self.DecisionMakerInteractor.get_optimiser()

    def compute_underlying_best(self):

        true_underlying_utility = self.get_true_utility_function()
        weight = self.get_true_parameters()
        optimiser = self.get_optimiser()

        def top_true_utility(X):
            X = np.atleast_2d(X)
            Y_recommended, cost_new = self.objective.evaluate(X)
            Y_recommended = np.concatenate(Y_recommended, axis=1)
            uval = true_underlying_utility(y=Y_recommended,
                                           weights=weight[1],
                                           parameters=weight[0])

            return -np.array(uval).reshape(-1)

        sampled_uvals = top_true_utility(self.X)
        best_sampled_x = np.atleast_2d(self.X[np.argmin(sampled_uvals), :])
        true_best_x, true_best_val = optimiser(f=top_true_utility, include_point=best_sampled_x)

        best_y, _ = self.objective.evaluate(true_best_x)

        return true_best_x, -true_best_val

    def evaluate_objective(self):
        """
        Evaluates the objective
        """
        print(1)
        print(self.suggested_sample)
        self.Y_new, cost_new = self.objective.evaluate(self.suggested_sample)

        for j in range(self.objective.output_dim):
            print(self.Y_new[j])
            self.Y[j] = np.vstack((self.Y[j], self.Y_new[j]))

    def compute_current_best(self):
        current_acqX = self.acquisition.current_compute_acq()
        return current_acqX

    def _distance_last_evaluations(self):
        """
        Computes the distance between the last two evaluations.
        """
        return np.sqrt(sum((self.X[self.X.shape[0] - 1, :] - self.X[self.X.shape[0] - 2, :]) ** 2))

    def _compute_next_evaluations(self, pending_zipped_X=None, ignored_zipped_X=None, re_use=False):

        """
        Computes the location of the new evaluation (optimizes the acquisition in the standard case).
        :param pending_zipped_X: matrix of input configurations that are in a pending state (i.e., do not have an evaluation yet).
        :param ignored_zipped_X: matrix of input configurations that the user black-lists, i.e., those configurations will not be suggested again.
        :return:
        """
        ## --- Update the context if any

        self.acquisition.optimizer.context_manager = ContextManager(self.space, self.context)
        print("compute next evaluation")
        if self.sample_from_acq:
            print("suggest next location given THOMPSON SAMPLING")
            candidate_points = initial_design('latin', self.space, 2000)
            aux_var = self.acquisition._compute_acq(candidate_points)

        else:

            aux_var = self.evaluator.compute_batch(duplicate_manager=None, re_use=re_use, constrained=False)

        return self.space.zip_inputs(aux_var[0])

    def _update_model(self):
        """
        Updates the model (when more than one observation is available) and saves the parameters (if available).
        """
        if (self.num_acquisitions % self.model_update_interval) == 0:
            ### --- input that goes into the model (is unziped in case there are categorical variables)
            X_inmodel = self.space.unzip_inputs(self.X)
            Y_inmodel = list(self.Y)
            self.model.updateModel(X_inmodel, Y_inmodel)

        ### --- Save parameters of the model

    def get_evaluations(self):
        return self.X.copy(), self.Y.copy()

    def func_val(self, x):
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        Y, _ = self.objective.evaluate(x, true_val=True)
        C, _ = self.constraint.evaluate(x, true_val=True)
        Y = np.array(Y).reshape(-1)
        out = Y.reshape(-1) * np.product(np.concatenate(C, axis=1) < 0, axis=1).reshape(-1)
        out = np.array(out).reshape(-1)
        return -out
