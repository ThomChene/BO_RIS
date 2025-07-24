import numpy as np
import matplotlib.pyplot as plt
from Functions.Channel.scenarios import Scenario_builder
from Functions.Visualize.visualize import visualize_3D,visualize_beam,visualize_hist

####### Plots Histogram: Channel with NLOS, Codebook: with narrow beams #######
####### With Received signal strenghth in dB and without dB #######
####### Then plots 3D for 3 narrow beams
####### Plot beams

np.random.seed(20)

################# Parameters #################
size_data_train = 1000
size_data_test = 0

name_scenario = "RIS_Uplink_No_direct_link_codebook_hierarchical_5_paths"

################# Create Data #################
scenario_data = Scenario_builder(name_scenario = name_scenario,size_data_train = size_data_train,size_data_test = size_data_test)
scenario_data.read_scenario()
data_train,data_test,stats_data = scenario_data.generate()

################# Choose Feedback Model #################

for data in data_train:
    data.set_feedback_model(type_feedback = "Quadratic")

######## Visualize Data #########
filename_hist = "./Plots/Histogram_narrow_beam_NLOS.dat"
filename_3D = "./Plots/3D_plot_NLOS.dat"
filename_beam = "./Plots/shape_beams_NLOS.dat"

visualize_3D(data_train,[14,15,16],save=True,filename = filename_3D)
visualize_beam(data_train,[14,15,16],save=True,filename = filename_beam)
visualize_hist(data_train,14,save=True,filename = filename_hist)

################# Choose Feedback Model #################

for data in data_train:
    data.set_feedback_model(type_feedback = "dB")
    
######## Visualize Data #########
filename_hist_dB = "./Plots/Histogram_narrow_beam_dB_NLOS.dat"
filename_3D_dB = "./Plots/3D_plot_dB_NLOS.dat"
filename_beam_dB = "./Plots/shape_beams_dB_NLOS.dat"

visualize_3D(data_train,[14,15,16],save=True,filename = filename_3D_dB)
visualize_beam(data_train,[14,15,16],save=True,filename = filename_beam_dB)
visualize_hist(data_train,14,save=True,filename = filename_hist_dB)

plt.show()
        