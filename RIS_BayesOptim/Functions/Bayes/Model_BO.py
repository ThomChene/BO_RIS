import numpy as np

class Model_Bayesian_Optimization:
    """Contains the model:
    Mean_model = [Mean_mixture_1,...,Mean_mixture_M] for M mixtures
    each Mean_mixture_i = [Mean_value_1,...,Mean_value_N] for N elements in the codebook
    Conditional_Compressed_Covariance and Conditional_Compressed_Mean will change during
    """
    def __init__(self,type_model:str,Mean_model:list[np.ndarray],Covariance_model:list[np.ndarray],Mixture_coefficient:list[float]=[1],percent_pca:float=1):
        self.type_model = type_model
        self.Mean_model = Mean_model
        self.Covariance_model = Covariance_model
        self.Mixture_coefficient = Mixture_coefficient
        self.Conditional_Mixture_coefficient = Mixture_coefficient
        self.percent_pca = percent_pca
        self.compressed_version()
    
    def compressed_version(self):
        covariance = self.Covariance_model
        Compression_Covariance = []
        Compressed_Mean = []
        Compressed_Covariance = []
        for cov in covariance:
            u,s,v = np.linalg.svd(cov, hermitian=True)
            len_comp = int(self.percent_pca*len(s))
            #S = 0
            #sum_eigenvalue = sum(s**2)
            #for eigenvalue in range(0,len(s)):
                #if S>sum_eigenvalue*self.percent_pca:
                   # break
                #S = S + s[eigenvalue]**2
            R = u[:,0:len_comp]*np.sqrt(s[0:len_comp])
            Compression_Covariance.append(R)
            Compressed_Mean.append(np.zeros(len_comp))
            Compressed_Covariance.append(np.eye(len_comp))
        self.Compression_Covariance = Compression_Covariance
        
        self.Compressed_Mean = Compressed_Mean
        self.Compressed_Covariance = Compressed_Covariance
       
    def re_init(self):
        self.Conditional_Compressed_Mean = self.Compressed_Mean.copy()
        self.Conditional_Compressed_Covariance = self.Compressed_Covariance.copy()
        self.Conditional_Mixture_coefficient = self.Mixture_coefficient.copy()

    def get_type(self)->str:
        return self.type_model
    
    def get_moments_and_compressed(self)->tuple[list[np.ndarray],list[np.ndarray],list[np.ndarray],list[float],list[np.ndarray]]:
        return self.Mean_model,self.Conditional_Compressed_Mean,self.Conditional_Compressed_Covariance,self.Conditional_Mixture_coefficient,self.Compression_Covariance
    
    def update_moments(self,all_value_received:list,all_indices_tested:list):
        new_index_tested = all_indices_tested[-1]
        new_value_received = all_value_received[-1]
        likelihood_mixture = []
        for mix in range(0,len(self.Mixture_coefficient)):
            if self.Conditional_Mixture_coefficient[mix] == 0:
                likelihood_mixture.append(0)
            else:
                compression_cov = self.Compression_Covariance[mix]
                Compressed_Mean = self.Conditional_Compressed_Mean[mix]
                mean_value = np.dot(compression_cov[new_index_tested],Compressed_Mean) + self.Mean_model[mix][new_index_tested]

                Compressed_Covariance = self.Conditional_Compressed_Covariance[mix]
                Regression_Cov = np.dot(compression_cov[new_index_tested],Compressed_Covariance)           
                sigma_compressed = np.dot(Regression_Cov,compression_cov[new_index_tested].T)
                Centered_Compressed_mean = np.array(new_value_received-mean_value)
                #print("before")
                #print(Compressed_Mean)
                Conditional_Compressed_Mean = Regression_Cov*Centered_Compressed_mean/sigma_compressed + Compressed_Mean
                Conditional_Compressed_Covariance = Compressed_Covariance - np.dot(np.array([Regression_Cov]).T,np.array([Regression_Cov]))/sigma_compressed

                self.Conditional_Compressed_Mean[mix] = Conditional_Compressed_Mean
                self.Conditional_Compressed_Covariance[mix] = Conditional_Compressed_Covariance
                
                
                #y_mu = np.array(self.Mean_model[mix][all_indices_tested]-all_value_received)
                #Sigma_tested = self.Covariance_model[mix][np.ix_(all_indices_tested,all_indices_tested)]

                #Sigma_inv = np.linalg.lstsq(Sigma_tested,y_mu.T,rcond=None)

                #likelihood_mixture.append(np.exp(-np.dot(y_mu,Sigma_inv[0]))*self.Mixture_coefficient[mix])
                likelihood_mixture.append(np.exp(-Centered_Compressed_mean**2/sigma_compressed/2)*self.Conditional_Mixture_coefficient[mix])
        ##### Calcul de l'inverse pas très efficace ##### 

        Conditional_Mixture = [cm/sum(likelihood_mixture) for cm in likelihood_mixture]
        #print(Conditional_Mixture)
        self.Conditional_Mixture_coefficient = Conditional_Mixture