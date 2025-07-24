import numpy as np
import math
from Functions.Channel import data_class
from Functions.Bayes import Model_BO
from Functions.Bayes import acquisition_functions
from tqdm import tqdm

class BO:
    """
    Bayesian Optimization:
    Composed of:
    A Model:
    - A gaussian multivariate
    - A mixture of gaussian
    An acquisition function:
    - EI
    - UCB
    """
    def __init__(self,Model:Model_BO.Model_Bayesian_Optimization,Acquisition:acquisition_functions.Acquisition_function):
        self.Model = Model 
        self.Acquisition = Acquisition
        
    def test(self,data_test:list[data_class.data_element],Number_uses_acquisition:int):
        print('Start Model: ' + self.Model.get_type() + ' with acquisition: ' + self.Acquisition.get_type())
        average_ratio_at_each_query:np.ndarray = np.array([0] * (Number_uses_acquisition+1))
        for i in tqdm(np.arange(0,len(data_test))):
            data:data_class.data_element = data_test[i]
            ### The points we receive ###
            All_points_received = []
            Indices_tested = []
            ratio_at_each_query = []
            current_max = -100 # Causes computation issues if -inf
           
            self.Model.re_init()

            Mean,Conditional_Compressed_Mean,Conditional_Compressed_Covariance,Mixture_coefficient,Compression_matrix = self.Model.get_moments_and_compressed()
            #print(Mean)
            current_argmax = self.Acquisition.most_likely(Mean,Conditional_Compressed_Mean,Compression_matrix,Mixture_coefficient)
            ratio_at_each_query.append(data.get_ratio_element_value_dB(current_argmax))
            
            for query in range(0,Number_uses_acquisition):
                index_to_test = self.Acquisition.step_acquisition(Mean,Conditional_Compressed_Mean,Conditional_Compressed_Covariance,Mixture_coefficient,Compression_matrix,current_max)
                #print("Elem")
                #print(data.get_element())
                #print(np.diag(np.dot(Compression_matrix[0],np.dot(Conditional_Compressed_Covariance[0],Compression_matrix[0].T))))
                #print(np.dot(Compression_matrix[0],Conditional_Compressed_Mean[0]) + Mean[0])
                value_received = data.get_element_value_index(index_to_test,noise = True)
                All_points_received.append(value_received)
                current_max = np.max(All_points_received)
                Indices_tested.append(index_to_test)
                self.Model.update_moments(All_points_received,Indices_tested)
                Mean,Conditional_Compressed_Mean,Conditional_Compressed_Covariance,Mixture_coefficient,Compression_matrix = self.Model.get_moments_and_compressed()
                
                current_argmax = self.Acquisition.most_likely(Mean,Conditional_Compressed_Mean,Compression_matrix,Mixture_coefficient,All_points_received,Indices_tested)
                ratio_at_each_query.append(data.get_ratio_element_value_dB(current_argmax))
                #print(Indices_tested)
                
            average_ratio_at_each_query = average_ratio_at_each_query + np.array(ratio_at_each_query) 
        return average_ratio_at_each_query/len(data_test)
        