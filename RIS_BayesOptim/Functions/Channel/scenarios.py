import math

from Functions.Channel import data_class,channel_models,codebook_types,signal_function,generate_data

#### Contains different scerarios ###

class Scenario_builder:
    """Contains the scenario from which we will generate the data:
    Number of antennas,RIS,LOS..."""
    def __init__(self,name_scenario:str,size_data_train:int,size_data_test:int):
        self.name_scenario = name_scenario
        self.size_data_train = size_data_train
        self.size_data_test = size_data_test
      
    def get_size(self)->int:
        return self.size_data
    
    def generate(self)->tuple[list[data_class.data_element],list[data_class.data_element],list]:
        if self.name_scenario == "Gaussian_Multivariate" or self.name_scenario == "Mixture_Gaussian_Multivariate":
            data_train,data_test,stats_data = generate_data.generate_data_synthetic(Law_synthetic = self.name_scenario,size_data_train = self.size_data_train,size_data_test = self.size_data_test,dimension_data_parameters = self.size_data)
            
        else:
            data_train,data_test = generate_data.generate_data_channel(size_data_train = self.size_data_train,size_data_test = self.size_data_test,scenario = "RIS_Uplink",codebooks = self.codebook,channels = self.channel,signal = self.signal)
            stats_data = ['None']
            
        return data_train,data_test,stats_data
    def read_scenario(self):
        print("Creating Data")
        if self.name_scenario == "Gaussian_Multivariate":
            print("For a gaussian multivariate law")
            dimension_data_parameters = 100 # Size gaussian vectors
            self.size_data = dimension_data_parameters
        
        if self.name_scenario == "Mixture_Gaussian_Multivariate":
            print("For a mixture of gaussian multivariate law")
            dimension_data_parameters = 100 # Size gaussian vectors
            self.n_mixture = 5 # How many components in the mixture
            self.size_data = dimension_data_parameters
        
        if self.name_scenario == "RIS_Uplink":
            print("With a RIS uplink")
            N_UE = 1
            N_BS = 64
            N_RIS = 100
            
            ##### Channels #####
            number_paths_RIS_BS = 1
            ratio_LOS_NLOS_RIS_BS = 2
            ratio_sigma_RIS_BS = 0
            range_angles_RIS_BS = [((-math.pi/2,math.pi/2),(-math.pi/2,math.pi/2)) for l in range(0,number_paths_RIS_BS)]
        
            channel_RIS_BS = channel_models.geometric_far_field_channel_model(dimension_matrix = (N_BS,N_RIS),number_paths = number_paths_RIS_BS,ratio_LOS_NLOS = ratio_LOS_NLOS_RIS_BS,range_angles = range_angles_RIS_BS,ratio_sigma=ratio_sigma_RIS_BS)
            
            number_paths_UE_RIS = 1
            ratio_LOS_NLOS_UE_RIS = 2
            ratio_sigma_UE_RIS  = 0.1
            range_angles_UE_RIS = [((-math.pi/2,math.pi/2),(-math.pi/2,math.pi/2)) for l in range(0,number_paths_UE_RIS)]
        
            channel_UE_RIS = channel_models.geometric_far_field_channel_model(dimension_matrix = (N_RIS,N_UE),number_paths = number_paths_UE_RIS,ratio_LOS_NLOS = ratio_LOS_NLOS_UE_RIS,range_angles = range_angles_UE_RIS,ratio_sigma=ratio_sigma_UE_RIS)
            
            number_paths_UE_BS = 1
            ratio_LOS_NLOS_UE_BS = 2
            ratio_sigma_UE_BS  = 0.1
            range_angles_UE_BS = [((-math.pi/2,math.pi/2),(-math.pi/2,math.pi/2)) for l in range(0,number_paths_UE_BS)]
        
            channel_UE_BS = channel_models.geometric_far_field_channel_model(dimension_matrix = (N_BS,N_UE),number_paths = number_paths_UE_BS,ratio_LOS_NLOS = ratio_LOS_NLOS_UE_BS,range_angles = range_angles_UE_BS,ratio_sigma=ratio_sigma_UE_BS)
            
            ##### Codebooks #####
            n_beams = 3
            range_beams = (-math.pi/2,math.pi/2)
            constraint = 'None'
            
            Codebook_RIS = codebook_types.Narrow_codebook(dimension_channel = N_RIS,number_narrow_beams = n_beams,range_beams = range_beams,constraint = constraint)
            self.size_data = n_beams
            self.codebook = [Codebook_RIS]
            self.channel = [channel_RIS_BS,channel_UE_RIS,channel_UE_BS]
            self.signal = signal_function.Signal_function(N_UE)
            
        if self.name_scenario == "RIS_Uplink_No_direct_link_codebook_hierarchical_1_path":
            print("With a RIS uplink No direct link")
            N_UE = 1
            N_BS = 64
            N_RIS = 64
            
            ##### Channels #####
            number_paths_RIS_BS = 1
            ratio_LOS_NLOS_RIS_BS = 2
            ratio_sigma_RIS_BS = 0
            range_angles_RIS_BS = [((math.pi/4,math.pi/4),(-math.pi/4,-math.pi/4)) for l in range(0,number_paths_RIS_BS)]
            #range_angles_RIS_BS = [((-math.pi,-math.pi+math.pi/10),(-math.pi,-math.pi+math.pi/10)) for l in range(0,number_paths_RIS_BS)]
            
            channel_RIS_BS = channel_models.geometric_far_field_channel_model(dimension_matrix = (N_BS,N_RIS),number_paths = number_paths_RIS_BS,ratio_LOS_NLOS = ratio_LOS_NLOS_RIS_BS,range_angles = range_angles_RIS_BS,ratio_sigma=ratio_sigma_RIS_BS)
            
            number_paths_UE_RIS = 1
            ratio_LOS_NLOS_UE_RIS = 1
            ratio_sigma_UE_RIS  = 0
            range_angles_UE_RIS = [((-math.pi/2,math.pi/2),(-math.pi/2,0)) for l in range(0,number_paths_UE_RIS)]
            #range_angles_UE_RIS = [((-math.pi,-math.pi+math.pi/10),(-math.pi,-math.pi+math.pi/10)) for l in range(0,number_paths_UE_RIS)]
            
            channel_UE_RIS = channel_models.geometric_far_field_channel_model(dimension_matrix = (N_RIS,N_UE),number_paths = number_paths_UE_RIS,ratio_LOS_NLOS = ratio_LOS_NLOS_UE_RIS,range_angles = range_angles_UE_RIS,ratio_sigma=ratio_sigma_UE_RIS)
            
            channel_UE_BS_zeros = channel_models.Channel_Model(dimension_matrix = (N_BS,N_UE))
            
            ##### Codebooks #####
            # Codebook to create:
            # First create wide beams 
            # Then narrow beams
            n_beams_wide_1 = 2
            width_factor_1 = 2
            n_beams_wide_2 = 4
            width_factor_2 = 4
            n_beams_wide_3 = 8
            width_factor_3 = 8
            n_beams_wide_4 = 16
            width_factor_4 = 16
            n_beams_wide_5 = 32
            width_factor_5 = 32
            n_beams_narrow = 64
            range_beams = (-math.pi,math.pi)
            constraint = 'None'
            
            Codebook_RIS_wide = codebook_types.Wide_codebook(dimension_channel = N_RIS,number_wide_beams = n_beams_wide_1,range_beams = range_beams,width_factor = width_factor_1,constraint = constraint)
            Codebook_RIS_wide_2 = codebook_types.Wide_codebook(dimension_channel = N_RIS,number_wide_beams = n_beams_wide_2,range_beams = range_beams,width_factor = width_factor_2,constraint = constraint)
            Codebook_RIS_wide.fuse_codebooks(Codebook_RIS_wide_2)
            Codebook_RIS_wide_3 = codebook_types.Wide_codebook(dimension_channel = N_RIS,number_wide_beams = n_beams_wide_3,range_beams = range_beams,width_factor = width_factor_3,constraint = constraint)
            Codebook_RIS_wide.fuse_codebooks(Codebook_RIS_wide_3)
            Codebook_RIS_wide_4 = codebook_types.Wide_codebook(dimension_channel = N_RIS,number_wide_beams = n_beams_wide_4,range_beams = range_beams,width_factor = width_factor_4,constraint = constraint)
            Codebook_RIS_wide.fuse_codebooks(Codebook_RIS_wide_4)
            Codebook_RIS_wide_5 = codebook_types.Wide_codebook(dimension_channel = N_RIS,number_wide_beams = n_beams_wide_5,range_beams = range_beams,width_factor = width_factor_5,constraint = constraint)
            Codebook_RIS_wide.fuse_codebooks(Codebook_RIS_wide_5)
            Codebook_RIS = codebook_types.Narrow_codebook(dimension_channel = N_RIS,number_narrow_beams = n_beams_narrow,range_beams = range_beams,constraint = constraint)
            Codebook_RIS_wide.fuse_codebooks(Codebook_RIS)
            
            self.size_data = n_beams_narrow + n_beams_wide_1 + n_beams_wide_2 + n_beams_wide_3 + n_beams_wide_4 + n_beams_wide_5
            self.codebook = [Codebook_RIS_wide]
            self.channel = [channel_RIS_BS,channel_UE_RIS,channel_UE_BS_zeros]
            self.signal = signal_function.Signal_function(N_UE)
           
        if self.name_scenario == "RIS_Uplink_No_direct_link_codebook_hierarchical_5_paths":
            print("With a RIS uplink No direct link")
            N_UE = 1
            N_BS = 64
            N_RIS = 64
            
            ##### Channels #####
            number_paths_RIS_BS = 1
            ratio_LOS_NLOS_RIS_BS = 2
            ratio_sigma_RIS_BS = 0
            range_angles_RIS_BS = [((math.pi/4,math.pi/4),(-math.pi/4,-math.pi/4)) for l in range(0,number_paths_RIS_BS)]
            #range_angles_RIS_BS = [((-math.pi,-math.pi+math.pi/10),(-math.pi,-math.pi+math.pi/10)) for l in range(0,number_paths_RIS_BS)]
            
            channel_RIS_BS = channel_models.geometric_far_field_channel_model(dimension_matrix = (N_BS,N_RIS),number_paths = number_paths_RIS_BS,ratio_LOS_NLOS = ratio_LOS_NLOS_RIS_BS,range_angles = range_angles_RIS_BS,ratio_sigma=ratio_sigma_RIS_BS)
            
            number_paths_UE_RIS = 5
            ratio_LOS_NLOS_UE_RIS = 10
            ratio_sigma_UE_RIS  = 0
            range_angles_UE_RIS = [((-math.pi/2,math.pi/2),(-math.pi/2,0)) for l in range(0,number_paths_UE_RIS)]
            #range_angles_UE_RIS = [((-math.pi,-math.pi+math.pi/10),(-math.pi,-math.pi+math.pi/10)) for l in range(0,number_paths_UE_RIS)]
            
            channel_UE_RIS = channel_models.geometric_far_field_channel_model(dimension_matrix = (N_RIS,N_UE),number_paths = number_paths_UE_RIS,ratio_LOS_NLOS = ratio_LOS_NLOS_UE_RIS,range_angles = range_angles_UE_RIS,ratio_sigma=ratio_sigma_UE_RIS)
            
            channel_UE_BS_zeros = channel_models.Channel_Model(dimension_matrix = (N_BS,N_UE))
            
            ##### Codebooks #####
            # Codebook to create:
            # First create wide beams 
            # Then narrow beams
            n_beams_wide_1 = 2
            width_factor_1 = 2
            n_beams_wide_2 = 4
            width_factor_2 = 4
            n_beams_wide_3 = 8
            width_factor_3 = 8
            n_beams_wide_4 = 16
            width_factor_4 = 16
            n_beams_wide_5 = 32
            width_factor_5 = 32
            n_beams_narrow = 64
            range_beams = (-math.pi,math.pi)
            constraint = 'None'
            
            Codebook_RIS_wide = codebook_types.Wide_codebook(dimension_channel = N_RIS,number_wide_beams = n_beams_wide_1,range_beams = range_beams,width_factor = width_factor_1,constraint = constraint)
            Codebook_RIS_wide_2 = codebook_types.Wide_codebook(dimension_channel = N_RIS,number_wide_beams = n_beams_wide_2,range_beams = range_beams,width_factor = width_factor_2,constraint = constraint)
            Codebook_RIS_wide.fuse_codebooks(Codebook_RIS_wide_2)
            Codebook_RIS_wide_3 = codebook_types.Wide_codebook(dimension_channel = N_RIS,number_wide_beams = n_beams_wide_3,range_beams = range_beams,width_factor = width_factor_3,constraint = constraint)
            Codebook_RIS_wide.fuse_codebooks(Codebook_RIS_wide_3)
            Codebook_RIS_wide_4 = codebook_types.Wide_codebook(dimension_channel = N_RIS,number_wide_beams = n_beams_wide_4,range_beams = range_beams,width_factor = width_factor_4,constraint = constraint)
            Codebook_RIS_wide.fuse_codebooks(Codebook_RIS_wide_4)
            Codebook_RIS_wide_5 = codebook_types.Wide_codebook(dimension_channel = N_RIS,number_wide_beams = n_beams_wide_5,range_beams = range_beams,width_factor = width_factor_5,constraint = constraint)
            Codebook_RIS_wide.fuse_codebooks(Codebook_RIS_wide_5)
            Codebook_RIS = codebook_types.Narrow_codebook(dimension_channel = N_RIS,number_narrow_beams = n_beams_narrow,range_beams = range_beams,constraint = constraint)
            Codebook_RIS_wide.fuse_codebooks(Codebook_RIS)
            
            self.size_data = n_beams_narrow + n_beams_wide_1 + n_beams_wide_2 + n_beams_wide_3 + n_beams_wide_4 + n_beams_wide_5
            self.codebook = [Codebook_RIS_wide]
            self.channel = [channel_RIS_BS,channel_UE_RIS,channel_UE_BS_zeros]
            self.signal = signal_function.Signal_function(N_UE)
             
        if self.name_scenario == "RIS_Uplink_No_direct_link_codebook_hierarchical_1_path_quantized":
            print("With a RIS uplink No direct link")
            N_UE = 1
            N_BS = 64
            N_RIS = 64
            
            ##### Channels #####
            number_paths_RIS_BS = 1
            ratio_LOS_NLOS_RIS_BS = 2
            ratio_sigma_RIS_BS = 0
            range_angles_RIS_BS = [((math.pi/4,math.pi/4),(-math.pi/4,-math.pi/4)) for l in range(0,number_paths_RIS_BS)]
            #range_angles_RIS_BS = [((-math.pi,-math.pi+math.pi/10),(-math.pi,-math.pi+math.pi/10)) for l in range(0,number_paths_RIS_BS)]
            
            channel_RIS_BS = channel_models.geometric_far_field_channel_model(dimension_matrix = (N_BS,N_RIS),number_paths = number_paths_RIS_BS,ratio_LOS_NLOS = ratio_LOS_NLOS_RIS_BS,range_angles = range_angles_RIS_BS,ratio_sigma=ratio_sigma_RIS_BS)
            
            number_paths_UE_RIS = 1
            ratio_LOS_NLOS_UE_RIS = 1
            ratio_sigma_UE_RIS  = 0
            range_angles_UE_RIS = [((-math.pi/2,math.pi/2),(-math.pi/2,0)) for l in range(0,number_paths_UE_RIS)]
            #range_angles_UE_RIS = [((-math.pi,-math.pi+math.pi/10),(-math.pi,-math.pi+math.pi/10)) for l in range(0,number_paths_UE_RIS)]
            
            channel_UE_RIS = channel_models.geometric_far_field_channel_model(dimension_matrix = (N_RIS,N_UE),number_paths = number_paths_UE_RIS,ratio_LOS_NLOS = ratio_LOS_NLOS_UE_RIS,range_angles = range_angles_UE_RIS,ratio_sigma=ratio_sigma_UE_RIS)
            
            channel_UE_BS_zeros = channel_models.Channel_Model(dimension_matrix = (N_BS,N_UE))
            
            ##### Codebooks #####
            # Codebook to create:
            # First create wide beams 
            # Then narrow beams
            n_beams_wide_1 = 2
            width_factor_1 = 2
            n_beams_wide_2 = 4
            width_factor_2 = 4
            n_beams_wide_3 = 8
            width_factor_3 = 8
            n_beams_wide_4 = 16
            width_factor_4 = 16
            n_beams_wide_5 = 32
            width_factor_5 = 32
            n_beams_narrow = 64
            range_beams = (-math.pi,math.pi)
            constraint = 'Quantized'
            n_bits=1
            
            Codebook_RIS_wide = codebook_types.Wide_codebook(dimension_channel = N_RIS,number_wide_beams = n_beams_wide_1,range_beams = range_beams,width_factor = width_factor_1,constraint = constraint,n_bits = n_bits)
            Codebook_RIS_wide_2 = codebook_types.Wide_codebook(dimension_channel = N_RIS,number_wide_beams = n_beams_wide_2,range_beams = range_beams,width_factor = width_factor_2,constraint = constraint,n_bits = n_bits)
            Codebook_RIS_wide.fuse_codebooks(Codebook_RIS_wide_2)
            Codebook_RIS_wide_3 = codebook_types.Wide_codebook(dimension_channel = N_RIS,number_wide_beams = n_beams_wide_3,range_beams = range_beams,width_factor = width_factor_3,constraint = constraint,n_bits = n_bits)
            Codebook_RIS_wide.fuse_codebooks(Codebook_RIS_wide_3)
            Codebook_RIS_wide_4 = codebook_types.Wide_codebook(dimension_channel = N_RIS,number_wide_beams = n_beams_wide_4,range_beams = range_beams,width_factor = width_factor_4,constraint = constraint,n_bits = n_bits)
            Codebook_RIS_wide.fuse_codebooks(Codebook_RIS_wide_4)
            Codebook_RIS_wide_5 = codebook_types.Wide_codebook(dimension_channel = N_RIS,number_wide_beams = n_beams_wide_5,range_beams = range_beams,width_factor = width_factor_5,constraint = constraint,n_bits = n_bits)
            Codebook_RIS_wide.fuse_codebooks(Codebook_RIS_wide_5)
            Codebook_RIS = codebook_types.Narrow_codebook(dimension_channel = N_RIS,number_narrow_beams = n_beams_narrow,range_beams = range_beams,constraint = constraint,n_bits = n_bits)
            Codebook_RIS_wide.fuse_codebooks(Codebook_RIS)
            
            self.size_data = n_beams_narrow + n_beams_wide_1 + n_beams_wide_2 + n_beams_wide_3 + n_beams_wide_4 + n_beams_wide_5
            self.codebook = [Codebook_RIS_wide]
            self.channel = [channel_RIS_BS,channel_UE_RIS,channel_UE_BS_zeros]
            self.signal = signal_function.Signal_function(N_UE)
            
        if self.name_scenario == "RIS_Uplink_No_direct_link_codebook_random":
            print("With a RIS uplink No direct link")
            N_UE = 1
            N_BS = 64
            N_RIS = 64
            
            ##### Channels #####
            number_paths_RIS_BS = 1
            ratio_LOS_NLOS_RIS_BS = 2
            ratio_sigma_RIS_BS = 0
            range_angles_RIS_BS = [((math.pi/4,math.pi/4),(-math.pi/4,-math.pi/4)) for l in range(0,number_paths_RIS_BS)]
            #range_angles_RIS_BS = [((-math.pi,-math.pi+math.pi/10),(-math.pi,-math.pi+math.pi/10)) for l in range(0,number_paths_RIS_BS)]
            
            channel_RIS_BS = channel_models.geometric_far_field_channel_model(dimension_matrix = (N_BS,N_RIS),number_paths = number_paths_RIS_BS,ratio_LOS_NLOS = ratio_LOS_NLOS_RIS_BS,range_angles = range_angles_RIS_BS,ratio_sigma=ratio_sigma_RIS_BS)
            
            number_paths_UE_RIS = 1
            ratio_LOS_NLOS_UE_RIS = 10
            ratio_sigma_UE_RIS  = 0
            range_angles_UE_RIS = [((-math.pi/2,math.pi/2),(-math.pi/2,0)) for l in range(0,number_paths_UE_RIS)]
            #range_angles_UE_RIS = [((-math.pi,-math.pi+math.pi/10),(-math.pi,-math.pi+math.pi/10)) for l in range(0,number_paths_UE_RIS)]
            
            channel_UE_RIS = channel_models.geometric_far_field_channel_model(dimension_matrix = (N_RIS,N_UE),number_paths = number_paths_UE_RIS,ratio_LOS_NLOS = ratio_LOS_NLOS_UE_RIS,range_angles = range_angles_UE_RIS,ratio_sigma=ratio_sigma_UE_RIS)
            
            channel_UE_BS_zeros = channel_models.Channel_Model(dimension_matrix = (N_BS,N_UE))
            
            ##### Codebooks #####
            range_beams = (-math.pi,math.pi)
            constraint = 'None'
            n_beams = 100

            Codebook_RIS = codebook_types.Random_codebook(dimension_channel = N_RIS,number_beams = n_beams,constraint = constraint)
            
            self.size_data = n_beams
            
            self.codebook = [Codebook_RIS]
            self.channel = [channel_RIS_BS,channel_UE_RIS,channel_UE_BS_zeros]
            self.signal = signal_function.Signal_function(N_UE)
            

    
    