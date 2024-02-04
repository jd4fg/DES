"""
Code that contains the ModelSearch class, which is used for searching the best model for a given dataset.
"""
    
# 3rd party imports
from model import Model
import pandas as pd
from matplotlib import pyplot as plt
        

class ModelSearch:
    def __init__(self, window_size_list: list, default_k_list: list, rate_of_change_list: list,
                 significance_level_list: list, forced_decrease: bool) -> None:
        """
        Initializes model search space.
        A list of values for each parameter is given, and all combinations of these values are used to create models.
        
        :param list window_size_list: a list of size of the time window to use for the models
        :param list default_k_list: a list of values of k to use for the models before the initialization phase is over. Has to be in [0, 1]
        :param list rate_of_change_list: a list of maximum amount of change in k per time step. Has to be in [0, 1]
        :param list significance_level_list: a list of significance levels for the p-value of the t-test. Has to be in (0, 1)
        :param bool forced_decrease: whether the model should be forced to decrease k if the mean error is not significantly different from 0, defaults to False
            True is equal to Method 1 in the paper, False is equal to Method 2 in the paper, defaults to Method 1
        """
        
        # saves the search space
        self.window_size_list = window_size_list
        self.default_k_list = default_k_list
        self.rate_of_change_list = rate_of_change_list
        self.significance_level_list = significance_level_list
        self.forced_decrease = forced_decrease
        
        # initializes list of models and their evaluations
        self.models = []
        self.evaluations = pd.DataFrame(columns=["model_idx", "window_size", "default_k", "rate_of_change", "significance_level",
                                                 "abs_second_derivative", "abs_error"])
        
        # define constants for sorting
        self.BY_SECOND_DERIVATIVE = "abs_second_derivative"
        self.BY_ERROR = "abs_error"
        
    def search_models(self, data: list, standardize: bool = True):
        """
        Evaluates all models for the given data.
        
        :param list data: data to search for the best model
        :param bool standardize: whether to standardize the evaluation metrics, defaults to True
            Standardization is done by dividing the metric by the maximum value of the metric
        """
        
        # re-initializes the list of models and their evaluations
        self.models = []
        self.evaluations = pd.DataFrame(columns=["model_idx", "window_size", "default_k", "rate_of_change", "significance_level",
                                                 "abs_second_derivative", "abs_error"])
        
        # for each parameter combination, create a model and evaluate it
        for window_size in self.window_size_list:
            for default_k in self.default_k_list:
                for rate_of_change in self.rate_of_change_list:
                    for significance_level in self.significance_level_list:
                        # create and execute model
                        model = Model(window_size, default_k, rate_of_change, significance_level, self.forced_decrease)
                        model.add_points(data)
                        
                        # get model evaluation metrics
                        second_derivative = model.evaluate_model_discrete_2nd_der()
                        abs_error = model.evaluate_model_error()
                        
                        # save model and evaluation metrics
                        self.models.append(model)
                        self.evaluations.loc[len(self.evaluations)] = [len(self.models)-1, window_size, default_k, rate_of_change, significance_level,
                                                                       second_derivative, abs_error]
        
        # standardize second derivative and absolute error
        if standardize:
            self.evaluations["abs_second_derivative"] = self.evaluations["abs_second_derivative"] / max(self.evaluations["abs_second_derivative"])
            self.evaluations["abs_error"] = self.evaluations["abs_error"] / max(self.evaluations["abs_error"])
                        
    def plot_metrics(self, ax: plt.Axes) -> None:
        """
        Plots the evaluation metrics for all models.
        
        :param plt.Axes ax: axes to plot the metrics on
        """
        # plot metrics
        ax.scatter(self.evaluations['abs_error'], self.evaluations['abs_second_derivative'], label="metrics")
        ax.legend()
        # set labels
        ax.set_xlabel('Error')
        ax.set_ylabel('Second derivative')
                    
    def get_top_models(self, num_models: int, by: str, constraint_by: str = None, constraint_val: float = None, plot: bool = False) -> dict:
        """
        Returns the top num_models models sorted by the given metric and constrained by another.
        
        :param int num_models: number of models to return
        :param str by: metric to sort by, has to be one of the constants defined in the class (self.BY_SECOND_DERIVATIVE, self.BY_ERROR)
        :param str constraint_by: metric to constrain by (ensures that only considers models that have a value less than or equal to constraint_val),
            has to be one of the constants defined in the class (self.BY_SECOND_DERIVATIVE, self.BY_ERROR) defaults to None
        :param float constraint_val: value to constrain by, defaults to None
        :param bool plot: whether to plot the top models, their k values and error distribution, defaults to False
        
        :return dict: dictionary containing the top models and their evaluations in the form of {'models': list of Model class, 'evaluations': pd.DataFrame}
        """
        
        # sort by given metric
        sorted_models = self.evaluations.sort_values(by=by, ascending=True)
        
        # removes models that do not satisfy the constraint
        if constraint_by is not None:
            sorted_models = sorted_models[sorted_models[constraint_by] <= constraint_val]
        
        # get the top models
        top_model_evals = sorted_models.head(num_models)
        top_model_idxs = top_model_evals["model_idx"].values.tolist()
        top_models = [self.models[int(i)] for i in top_model_idxs]
        
        # plot the models, their k values and error distributions
        if plot:
            _, axs = plt.subplots(num_models, 3, figsize=(15, 5*num_models))
            for i_row, row in enumerate(top_model_evals.iterrows()):
                er = round(row[1]['abs_error'], 2)
                secd = round(row[1]['abs_second_derivative'], 2)
                
                model = top_models[i_row]
                
                # plot model
                model.plot_model(axs[i_row, 0], er, secd)
                # plot the k values throughout the model
                model.plot_k(axs[i_row, 1])
                # plot error distributions
                model.plot_errors(axs[i_row, 2])
        
        return {'models': top_models, 'evaluations': top_model_evals}
    
    