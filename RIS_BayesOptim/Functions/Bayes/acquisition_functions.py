import numpy as np
from scipy.stats import norm
import math

class Acquisition_function:
    """To declare the next index to test:type_acquisition
    To declare which of the index is the best:type_choice
    """
    def __init__(self,type_acquisition:str,type_choice:str,hyperparameter_UCB:float=0):
        self.type_acquisition = type_acquisition
        self.type_choice = type_choice
        self.hyperparameter_UCB = hyperparameter_UCB
       
    def get_type(self)->str:
        return self.type_acquisition
    
    def most_likely(self,Mean:list[np.ndarray],Conditional_Compressed_Mean:list[np.ndarray],Compression_matrix:list[np.ndarray],mixture_coeff:list[float],All_points_received:list[float]=[],Indices_tested:list[int]=[])->int:
        if len(All_points_received) == 0:
            Score = self.acquisition_Mean(Mean,Conditional_Compressed_Mean,mixture_coeff,Compression_matrix)
            return np.argmax(Score)
        else:
            if self.type_choice == 'Mean':
                Score = self.acquisition_Mean(Mean,Conditional_Compressed_Mean,mixture_coeff,Compression_matrix)
                return np.argmax(Score)
            elif self.type_choice == 'Best_so_far':
                return Indices_tested[np.argmax(All_points_received)]
            else:
                return Indices_tested[np.argmax(All_points_received)]
    
    def step_acquisition(self,Mean:list[np.ndarray],Conditional_Compressed_Mean:list[np.ndarray],Covariance:list[np.ndarray],mixture_coeff:list[float],Compression_matrix:list[np.ndarray],current_max:float,query:int=0)->int:
        if self.type_acquisition == 'EI':
            Score = self.acquisition_EI(Mean,Conditional_Compressed_Mean,Covariance,mixture_coeff,Compression_matrix,current_max)
            
        if self.type_acquisition == 'UCB':
            Score = self.acquisition_UCB(Mean,Conditional_Compressed_Mean,Covariance,mixture_coeff,Compression_matrix)
           
        if self.type_acquisition == 'Mean':
            Score = self.acquisition_Mean(Mean,Conditional_Compressed_Mean,mixture_coeff,Compression_matrix)
        #print("score")
        #print(Score)
        return np.argmax(Score)
        
    def acquisition_EI(self,Mean:list[np.ndarray],Conditional_Compressed_Mean:list[np.ndarray],Covariance:list[np.ndarray],mixture:list[float],Compression_matrix:list[np.ndarray],current_max:float)->np.ndarray:
        EI = np.zeros(len(Mean[0]))
        for mix in range(0,len(mixture)):
            Conditional_Mean = np.dot(Compression_matrix[mix],Conditional_Compressed_Mean[mix]) + Mean[mix]
            Conditional_Covariance = np.dot(Compression_matrix[mix],np.dot(Covariance[mix],Compression_matrix[mix].T))
            diag_Covariance = np.diag(Conditional_Covariance)
            centered_mean = (Conditional_Mean-current_max)/(diag_Covariance+1e-10) # to prevent dividing by zero
            EI = EI + mixture[mix]*((Conditional_Mean-current_max)*norm.cdf(centered_mean) + diag_Covariance*norm.pdf(centered_mean))
        return EI
    
    def acquisition_UCB(self,Mean:list[np.ndarray],Conditional_Compressed_Mean:list[np.ndarray],Covariance:list[np.ndarray],mixture:list[float],Compression_matrix:list[np.ndarray],query:int=0)->np.ndarray:
        UCB = np.zeros(len(Mean[0]))
        beta =  self.hyperparameter_UCB
        for mix in range(0,len(mixture)):
            Conditional_Mean = np.dot(Compression_matrix[mix],Conditional_Compressed_Mean[mix]) + Mean[mix]
            Conditional_Covariance = np.dot(Compression_matrix[mix],np.dot(Covariance[mix],Compression_matrix[mix].T))
            diag_Covariance = np.diag(Conditional_Covariance)
            UCB = UCB + mixture[mix]*(Conditional_Mean + diag_Covariance*beta)
        return UCB
    
    def acquisition_Mean(self,Mean:list[np.ndarray],Conditional_Compressed_Mean:list[np.ndarray],mixture:list[float],Compression_matrix:list[np.ndarray])->np.ndarray:
        average_Mean = np.zeros(len(Mean[0]))
        for mix in range(0,len(mixture)):
            Conditional_Mean = np.dot(Compression_matrix[mix],Conditional_Compressed_Mean[mix]) + Mean[mix]
            average_Mean = average_Mean + mixture[mix]*Conditional_Mean[mix]
        return average_Mean