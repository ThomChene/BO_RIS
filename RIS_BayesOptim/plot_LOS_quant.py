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

# Liste trucs à faire
# Hierarchique
# Tester version quantifiée du hierarchique
# Play with compression

################# Parameters #################
size_data_train = 20000
size_data_test = 1000

name_scenario = "RIS_Uplink_No_direct_link_codebook_hierarchical_1_path_quantized"

################# Create Data #################
scenario_data = Scenario_builder(name_scenario = name_scenario,size_data_train = size_data_train,size_data_test = size_data_test)
scenario_data.read_scenario()

n_comp_1 = 1
assert scenario_data.get_size()*n_comp_1 < size_data_train, "not enough data to fit model"
n_comp_2 = 5
assert scenario_data.get_size()*n_comp_2 < size_data_train, "not enough data to fit model"
n_comp_3 = 20
assert scenario_data.get_size()*n_comp_3 < size_data_train, "not enough data to fit model"
n_comp_4 = 50
assert scenario_data.get_size()*n_comp_4 < size_data_train, "not enough data to fit model"
n_comp_5 = 100
assert scenario_data.get_size()*n_comp_5 < size_data_train, "not enough data to fit model"

data_train,data_test,stats_data = scenario_data.generate()
################# Choose Feedback Model #################
type_noise = "Gaussian"
mean_noise = 0
var_noise = 0

type_feedback = "dB"

for data in data_train:
    data.set_feedback_model(type_noise = type_noise,mean_noise = mean_noise,var_noise = var_noise,type_feedback = type_feedback)

for data in data_test:
    data.set_feedback_model(type_noise = type_noise,mean_noise = mean_noise,var_noise = var_noise,type_feedback = type_feedback)
    
################# Fit Data #################

law_to_fit = "Mixture_Gaussian_Multivariate_hand"# Finds a parameter that structure the data and use it as a hidden parameter for the mixture

Mean_model_sklearn_1,Covariance_model_sklearn_1,Mixture_coefficient_sklearn_1 = fit_model_on_data(data_train,law_to_fit,n_comp=n_comp_1)
Mean_model_sklearn_2,Covariance_model_sklearn_2,Mixture_coefficient_sklearn_2 = fit_model_on_data(data_train,law_to_fit,n_comp=n_comp_2)
Mean_model_sklearn_3,Covariance_model_sklearn_3,Mixture_coefficient_sklearn_3 = fit_model_on_data(data_train,law_to_fit,n_comp=n_comp_3)
Mean_model_sklearn_4,Covariance_model_sklearn_4,Mixture_coefficient_sklearn_4 = fit_model_on_data(data_train,law_to_fit,n_comp=n_comp_4)
Mean_model_sklearn_5,Covariance_model_sklearn_5,Mixture_coefficient_sklearn_5 = fit_model_on_data(data_train,law_to_fit,n_comp=n_comp_5)

################# Acquisition Function #################
Number_uses_acquisition = 30 # Number of times we will use the acquisition function
assert Number_uses_acquisition < scenario_data.get_size()+1, "Reduce the number of uses of the acquisition function"

model_bo_sk_1 = Model_Bayesian_Optimization(type_model = "Gaussian_Multivariate",Mean_model = Mean_model_sklearn_1,Covariance_model = Covariance_model_sklearn_1,Mixture_coefficient = Mixture_coefficient_sklearn_1,percent_pca = 1)
acquisition_ei = Acquisition_function(type_acquisition = "EI",type_choice='Best_so_far')
bo_no_mixtures_ei_sk_1 = BO(model_bo_sk_1,acquisition_ei)

model_bo_sk_2 = Model_Bayesian_Optimization(type_model = "Gaussian_Multivariate",Mean_model = Mean_model_sklearn_2,Covariance_model = Covariance_model_sklearn_2,Mixture_coefficient = Mixture_coefficient_sklearn_2,percent_pca = 1)
acquisition_ei = Acquisition_function(type_acquisition = "EI",type_choice='Best_so_far')
bo_no_mixtures_ei_sk_2 = BO(model_bo_sk_2,acquisition_ei)

model_bo_sk_3 = Model_Bayesian_Optimization(type_model = "Gaussian_Multivariate",Mean_model = Mean_model_sklearn_3,Covariance_model = Covariance_model_sklearn_3,Mixture_coefficient = Mixture_coefficient_sklearn_3,percent_pca = 1)
acquisition_ei = Acquisition_function(type_acquisition = "EI",type_choice='Best_so_far')
bo_no_mixtures_ei_sk_3 = BO(model_bo_sk_3,acquisition_ei)

model_bo_sk_4 = Model_Bayesian_Optimization(type_model = "Gaussian_Multivariate",Mean_model = Mean_model_sklearn_4,Covariance_model = Covariance_model_sklearn_4,Mixture_coefficient = Mixture_coefficient_sklearn_4,percent_pca = 1)
acquisition_ei = Acquisition_function(type_acquisition = "EI",type_choice='Best_so_far')
bo_no_mixtures_ei_sk_4 = BO(model_bo_sk_4,acquisition_ei)

model_bo_sk_5 = Model_Bayesian_Optimization(type_model = "Gaussian_Multivariate",Mean_model = Mean_model_sklearn_5,Covariance_model = Covariance_model_sklearn_5,Mixture_coefficient = Mixture_coefficient_sklearn_5,percent_pca = 1)
acquisition_ei = Acquisition_function(type_acquisition = "EI",type_choice='Best_so_far')
bo_no_mixtures_ei_sk_5 = BO(model_bo_sk_5,acquisition_ei)

################## Simulation ####################
filename = "./Plots/acquisition_LOS_with_number_of_steps_quantized.dat"

######## Use Different Acquisition functions #########
random_acquisition_stats = random_acquisition(data_test,Number_uses_acquisition)
ei_sk1_acquisition_stats = bo_no_mixtures_ei_sk_1.test(data_test=data_test,Number_uses_acquisition=Number_uses_acquisition)
ei_sk2_acquisition_stats = bo_no_mixtures_ei_sk_2.test(data_test=data_test,Number_uses_acquisition=Number_uses_acquisition)
ei_sk3_acquisition_stats = bo_no_mixtures_ei_sk_3.test(data_test=data_test,Number_uses_acquisition=Number_uses_acquisition)
ei_sk4_acquisition_stats = bo_no_mixtures_ei_sk_4.test(data_test=data_test,Number_uses_acquisition=Number_uses_acquisition)
ei_sk5_acquisition_stats = bo_no_mixtures_ei_sk_5.test(data_test=data_test,Number_uses_acquisition=Number_uses_acquisition)

hier_acquisition_stats = hierarchical_acquisition(data_test=data_test,Number_uses_acquisition=Number_uses_acquisition,len_tree=6)

plot_results((random_acquisition_stats,"Random"),(ei_sk1_acquisition_stats,"EI_SK_1"),(ei_sk2_acquisition_stats,"EI_SK_2"),(ei_sk3_acquisition_stats,"EI_SK_3"),(ei_sk4_acquisition_stats,"EI_SK_4"),(ei_sk5_acquisition_stats,"EI_SK_5"),(hier_acquisition_stats,"Hier"),save=True,filename=filename)

plt.show()
        