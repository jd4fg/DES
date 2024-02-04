"""
Code that contains the definition of the Dynamic Exponential Filtering model (DEF).
"""

# 3rd party imports
import scipy
import numpy as np
from matplotlib import pyplot as plt


class Model:
    def __init__(self, window_size: int, default_k: float, rate_of_change: float,
                 significance_level: float, forced_decrease: bool=False, debug: bool=False):
        """
        Initializes model parameters.
        
        DEBUG MODE: If debug mode is enabled, the model will print out additional information.

        :param int window_size: size of the time window to use for the model
        :param float default_k: value of k to use for the model before the initialization phase is over. Has to be in [0, 1]
        :param float rate_of_change: maximum amount of change in k per time step. Has to be in [0, 1]
        :param float significance_level: significance level for the p-value of the t-test. Has to be in (0, 1)
        :param bool forced_decrease: whether the model should be forced to decrease k if the mean error is not significantly different from 0, defaults to False
            True is equal to Method 1 in the paper, False is equal to Method 2 in the paper, defaults to Method 1
        :param bool debug: whether this model is run in debug mode, defaults to False
        """
        
        # set model parameters
        self.window_size = window_size
        self.default_k = default_k
        self.rate_of_change = rate_of_change
        self.significance_level = significance_level
        self.forced_decrease = forced_decrease
        # set debug flag (vocal execution)
        self.debug = debug
        
        # keep track of past k values and errors 
        self.past_k = []
        self.past_errors = []
        
        # store data points, and smooth data points
        self.data = []
        self.smooth_data = []
        
        # calculate critical region for 2-tailed t-test with degrees of freedom equal to window size - 1
        self.critical_region = scipy.stats.t.ppf(1 - self.significance_level / 2, self.window_size - 1)
        
        # print whether debug mode is on
        if self.debug:
            print("DEBUG MODE: ON")
            print("Critical region: {}".format(self.critical_region))
        
        # initialize k as the default k
        self.k = self.default_k
        # define current phase as initialization
        self.init_phase = True
        
        
    def add_point(self, point):
        """
        Adds a new point to the model.
        
        :param float point: new point to add to the model
        """
        if self.debug:
            print("Adding point: {}".format(point))
        
        # if the model is not in the initialization phase, calculate the new k
        if not self.init_phase:
            # get errors in last time window
            errors = [self.smooth_data[i] - self.data[i] for i in range(-self.window_size, 0)]
            mean_error = sum(errors) / self.window_size
            
            # test whether the mean error is significantly different from 0 by comparing to the critical region
            reject_null = abs(mean_error) > self.critical_region
            
            # if the mean error is not significantly different from 0, decrease k / do not change k
            if not reject_null:
                if self.debug:
                    print("Mean error likely 0")
                    if self.forced_decrease:
                        print("Forced decrease")
                
                if self.forced_decrease:
                    self.k = max(self.k - self.rate_of_change, 0)

            # if the mean error is significantly different from 0, change it to make the mean error closer to 0
            else:
                # get potential k values
                k_higher = min(self.k + self.rate_of_change, 1)
                k_lower = max(self.k - self.rate_of_change, 0)
                # calculate new smooth points based on these k values
                smooth_higher = k_higher * point + (1-k_higher) * self.smooth_data[-1]
                smooth_lower = k_lower * point + (1-k_lower) * self.smooth_data[-1]
                
                # choose appropriate k value based on whether the mean error is above or below 0
                if mean_error > 0:
                    self.k = k_higher if smooth_higher < smooth_lower else k_lower
                else:
                    self.k = k_higher if smooth_higher > smooth_lower else k_lower
                
                if self.debug:
                    print("Mean error not 0; chosen: {}".format("higher" if self.k == k_higher else "lower"))
                    
            # add error and p value, k to past values
            self.past_errors.append(mean_error)
            
            if self.debug:
                print("Mean error: {}".format(mean_error))
                print("Reject null hypothesis: {}".format(reject_null))
                print("New k: {}".format(self.k))
        elif self.debug:
            print("Initialization phase, new k is default k: {}".format(self.default_k))
            
        # calculate new smooth point
        new_point = self.k * point + (1-self.k) * self.smooth_data[-1] if len(self.smooth_data) > 0 else point
        # append new k
        self.past_k.append(self.k)
        
        # append points
        self.smooth_data.append(new_point)
        self.data.append(point)
        
        # check whether we are still initializing
        if len(self.data) == self.window_size:
            self.init_phase = False
            if self.debug:
                print("Initialization phase finished")
        
        return new_point
    
    def add_points(self, points):
        """
        Adds a list of points to the model.
        
        :param list points: list of points to add to the model
        """
        for point in points:
            self.add_point(point)
            
    def refit_model(self):
        """
        Refits the model using the current smooth data as input.
        """        
        # reset model
        self.__init__(self.window_size, self.default_k, self.rate_of_change, self.significance_level, self.debug)
        # add all points
        self.add_points(self.data)
    
    def evaluate_model_discrete_2nd_der(self):
        """
        Evaluates the model by calculating the mean absolute second derivative of the smooth data.
        Only considres data points past initialization phase.
        
        Retruns:
            float: mean absolute second derivative of the smooth data
        """
        # get 2nd order derivative approximations at each point after initialization phase
        central_2nd_der = np.array([self.smooth_data[i-1] - 2*self.smooth_data[i] + self.smooth_data[i+1]
                                    for i in range(self.window_size+2, len(self.smooth_data)-2)])
 
        # calculate mean of second derivatives
        agg_second_deriv = sum(abs(central_2nd_der)) / len(central_2nd_der)
        
        if self.debug:
            print("Mean absolute second derivative: {}".format(agg_second_deriv))
        
        return agg_second_deriv
    
    def evaluate_model_error(self):
        """
        Evaluates the model by calculating the absolute error between the smooth data
        and the real data.
        Only considres data points past initialization phase.
        
        Retruns:
            float: absolute mean error of the model
        """
        # get errors after initialization phase
        errors = [self.smooth_data[i] - self.data[i] for i in range(self.window_size+1, len(self.data))]
        # calculate absolute mean error
        abs_mean_error = abs(sum(errors) / len(errors))
        
        if self.debug:
            print("Absolute mean error: {}".format(abs_mean_error))
        
        return abs_mean_error
    
    def print_params(self):
        """
        Prints the parameters of the model.
        """
        print("Window size: {}".format(self.window_size))
        print("Default k: {}".format(self.default_k))
        print("Rate of change: {}".format(self.rate_of_change))
        print("Significance level: {}".format(self.significance_level))
        print("Critical region: {}".format(self.critical_region))
        print("Debug mode: {}".format(self.debug))
        print("Method: {}".format(2 if self.forced_decrease else 1))
        
    def plot_model(self, ax: plt.Axes, abs_mean_error: float = None, second_der: float = None):
        """
        Plots the model.
        
        :param plt.Axes ax: axes to plot the model on
        :param float abs_mean_error: absolute mean error of the model, defaults to None
        :param float second_der: mean absolute second derivative of the smooth data, defaults to None
        """
        ax.scatter([i for i in range(len(self.data))], self.data, label="Data")
        ax.plot([i for i in range(len(self.data))], self.smooth_data, label="Smooth data", c="orange")
        ax.set_title(f"ws: {self.window_size}, dk: {self.default_k}, rc: {self.rate_of_change}, sl: {self.significance_level}, er: {abs_mean_error}, secd: {second_der}")
        ax.legend()
        
    def plot_k(self, ax: plt.Axes):
        """
        Plots the k values of the model.
        
        :param plt.Axes ax: axes to plot the k values on
        """
        ax.plot([i for i in range(len(self.past_k))], self.past_k)
        ax.set_title("K values")
        
    def plot_errors(self, ax: plt.Axes):
        """
        Plots the errors of the model. The errors in the initialization stage are not plotted.
        
        :param plt.Axes ax: axes to plot the errors on
        """
        ax.plot([i + self.window_size for i in range(len(self.past_errors))], self.past_errors)
        ax.set_title("Errors")
        ax.axhline(y=0, color='r')
        ax.axhline(y=self.critical_region, color='g')
        ax.axhline(y=-self.critical_region, color='g')

