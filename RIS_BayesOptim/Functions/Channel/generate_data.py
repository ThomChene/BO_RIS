import numpy as np
from tqdm import tqdm
from Functions.Channel import channel_models, codebook_types, data_class, signal_function

#######################################################

def generate_data_synthetic(Law_synthetic:str,size_data_train:int,size_data_test:int,dimension_data_parameters:int,n_mixture:int=1)->tuple[list,list,list]:
    """To generate Data from a known distribution for example a Gaussian
    Has nothing to do with telecommunications"""
    data_train:list = []
    data_test:list = []
    Mean:list = []
    Sigma:list = []
    ## Classic Multivariate Gaussian ##
    if Law_synthetic == "Gaussian_Multivariate":
        Mean.append(np.ones(dimension_data_parameters))
        #Mean.append(np.random.rand(dimension_data_parameters))
        sigma = 2*np.array(1/2-np.random.rand(dimension_data_parameters,dimension_data_parameters))
        sigma = np.dot(sigma.T,sigma) # The covariance needs to be symmetric
        Sigma.append(sigma)
        data_train_numpy = np.random.multivariate_normal(Mean[0],Sigma[0],size_data_train)
        data_test_numpy = np.random.multivariate_normal(Mean[0],Sigma[0],size_data_test)

        stats_data = [Law_synthetic,Mean,Sigma,np.array(1)]

        for data in data_train_numpy:
            data_train.append(data_class.data_element(data))

        for data in data_test_numpy:
            data_test.append(data_class.data_element(data))
            
    ## Classic Multivariate Gaussian ##
    elif Law_synthetic == "Mixture_Gaussian_Multivariate":
        size_data_train
        size_data_test
        mix_coeff = np.random.rand(n_mixture)
        mix_coeff = mix_coeff/sum(mix_coeff)
        size_data_train_mix = [int(mix_coeff[mix]*size_data_train) for mix in range(0,n_mixture)]
        size_data_test_mix = [int(mix_coeff[mix]*size_data_test) for mix in range(0,n_mixture)]
        for mix in range(0,n_mixture):
            Mean_mix = np.random.rand(dimension_data_parameters)
            #Mean = np.random.rand(number_indices)
            Sigma_mix = 2*np.array(1/2-np.random.rand(dimension_data_parameters,dimension_data_parameters))
            Sigma_mix = np.dot(Sigma_mix.T,Sigma_mix) # The covariance needs to be symmetric
            data_train_numpy = np.random.multivariate_normal(Mean_mix,Sigma_mix,size_data_train_mix[mix])
            data_test_numpy = np.random.multivariate_normal(Mean_mix,Sigma_mix,size_data_test_mix[mix])
            
            Mean.append(Mean_mix)
            Sigma.append(Sigma_mix)
            
            for data in data_train_numpy:
                data_train.append(data_class.data_element(data))

            for data in data_test_numpy:
                data_test.append(data_class.data_element(data))
        
        stats_data = [Law_synthetic,Mean,Sigma,mix_coeff]
            
    return data_train,data_test,stats_data

#######################################################

def generate_data_channel(size_data_train:int,size_data_test:int,scenario:str,codebooks:list[codebook_types.Codebook],channels:list[channel_models.Channel_Model],signal:signal_function.Signal_function)->tuple[list,list]:
    """To generate Data with channel"""   
    data_train:list = []
    data_test:list = []
    
    ##### RIS Scenario Uplink #####
    for i in tqdm(np.arange(0,size_data_train+size_data_test)):
        if scenario == 'RIS_Uplink':
            ### Channel is (H_RIS_BS Phi_RIS h_UE_RIS + H_UE_BS) ###
            H_RIS_BS,param_RIS_BS = channels[0].create_channel()
            h_UE_RIS,param_UE_RIS = channels[1].create_channel()
            h_UE_BS,param_UE_BS = channels[2].create_channel()
            Phi_RIS = codebooks[0].get_codewords()
            s = signal.generate()
            RSP = []
            for phi in Phi_RIS:
                H_1 = np.dot(H_RIS_BS,np.diag(phi))
                H_2 = np.dot(H_1,h_UE_RIS)
                H_3 = H_2 + h_UE_BS
                Hs = np.dot(H_3,s)
                rsp = sum(abs(Hs)**2)
                RSP.append(rsp)
        
        RSP_array = np.array(RSP)
        if i<size_data_train:
            #data_train.append(data_class.data_element(RSP_array,hidden=param_RIS_BS[0]+param_RIS_BS[1]+param_UE_RIS[0]+param_UE_RIS[1]))
            data_train.append(data_class.data_element(RSP_array,hidden=param_UE_RIS[0]+param_RIS_BS[0]+param_RIS_BS[1]))
        else:
            data_test.append(data_class.data_element(RSP_array))
                 
    return data_train,data_test
                    