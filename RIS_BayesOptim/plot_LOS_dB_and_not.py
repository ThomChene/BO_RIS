import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from Functions.Channel.scenarios import Scenario_builder
from Functions.Bayes.fitting import fit_model_on_data
from Functions.Bayes.random_acquisition import random_acquisition
from Functions.Visualize.visualize import visualize_3D,plot_results,visualize_beam,visualize_hist
from Functions.Bayes.Model_BO import Model_Bayesian_Optimization
from Functions.Bayes.bayesian_optimisation import BO
from Functions.Bayes.acquisition_functions import Acquisition_function
from Functions.Bayes.hierarchical import hierarchical_acquisition

np.random.seed(20)

################# Parameters #################
size_data_train = 10000
size_data_test = 5000

name_scenario = "RIS_Uplink_No_direct_link_codebook_hierarchical_1_path"

################# Create Data #################
scenario_data = Scenario_builder(name_scenario = name_scenario,size_data_train = size_data_train,size_data_test = size_data_test)
scenario_data.read_scenario()
data_train,data_test,stats_data = scenario_data.generate()

################# Choose Feedback Model #################

Number_uses_acquisition = 30 # Number of times we will use the acquisition function
assert Number_uses_acquisition < scenario_data.get_size()+1, "Reduce the number of uses of the acquisition function"

for data in data_train:
    data.set_feedback_model(type_feedback = "Quadratic")
for data in data_test:
    data.set_feedback_model(type_feedback = "Quadratic")

law_to_fit = "Mixture_Gaussian_Multivariate_hand"# Finds a parameter that structure the data and use it as a hidden parameter for the mixture
Mean_model_sklearn_1,Covariance_model_sklearn_1,Mixture_coefficient_sklearn_1 = fit_model_on_data(data_train,law_to_fit)

model_bo_sk_1 = Model_Bayesian_Optimization(type_model = "Gaussian_Multivariate",Mean_model = Mean_model_sklearn_1,Covariance_model = Covariance_model_sklearn_1,Mixture_coefficient = Mixture_coefficient_sklearn_1,percent_pca = 1)
acquisition_ei = Acquisition_function(type_acquisition = "EI",type_choice='Best_so_far')
bo_no_mixtures_ei_sk_1 = BO(model_bo_sk_1,acquisition_ei)
ei_sk1_acquisition_stats = bo_no_mixtures_ei_sk_1.test(data_test=data_test,Number_uses_acquisition=Number_uses_acquisition)

for data in data_train:
    data.set_feedback_model(type_feedback = "dB")
for data in data_test:
    data.set_feedback_model(type_feedback = "dB")

law_to_fit = "Mixture_Gaussian_Multivariate_hand"# Finds a parameter that structure the data and use it as a hidden parameter for the mixture
Mean_model_sklearn_2,Covariance_model_sklearn_2,Mixture_coefficient_sklearn_2 = fit_model_on_data(data_train,law_to_fit)

model_bo_sk_2 = Model_Bayesian_Optimization(type_model = "Gaussian_Multivariate",Mean_model = Mean_model_sklearn_2,Covariance_model = Covariance_model_sklearn_2,Mixture_coefficient = Mixture_coefficient_sklearn_2,percent_pca = 1)
acquisition_ei = Acquisition_function(type_acquisition = "EI",type_choice='Best_so_far')
bo_no_mixtures_ei_sk_2 = BO(model_bo_sk_2,acquisition_ei)
ei_sk2_acquisition_stats = bo_no_mixtures_ei_sk_2.test(data_test=data_test,Number_uses_acquisition=Number_uses_acquisition)
    


################## Simulation ####################
filename = "./Plots/acquisition_LOS_with_number_of_steps_dB_or_not.dat"

######## Use Different Acquisition functions #########
random_acquisition_stats = random_acquisition(data_test,Number_uses_acquisition)

hier_acquisition_stats = hierarchical_acquisition(data_test=data_test,Number_uses_acquisition=Number_uses_acquisition,len_tree=6)

plot_results((random_acquisition_stats,"Random"),(ei_sk1_acquisition_stats,"EI_SK_1"),(ei_sk2_acquisition_stats,"EI_SK_2"),save=True,filename=filename)

plt.show()
        