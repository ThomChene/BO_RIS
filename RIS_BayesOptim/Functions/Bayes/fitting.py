import numpy as np
import sklearn
from Functions.Channel import data_class

def fit_model_on_data(data_train:list[data_class.data_element],law_to_fit,n_comp=1)->tuple[list[np.ndarray],list[np.ndarray],list]:
    """Input: data to fit
    Output: law fitted on data"""
    Data = []
    for data_values in data_train:
        Data.append(data_values.get_element())
    if law_to_fit == "Gaussian_Multivariate":
        print("Fitting on a Gaussian Multivariate law")
        Mean = np.mean(Data, axis=0)
        Sigma = np.cov(Data, rowvar=0)
        mixture = [1]
        
    if law_to_fit == "Mixture_Gaussian_Multivariate_sklearn_em":
        print("Fitting on a Mixture of Gaussian Multivariate law")
        GM = sklearn.mixture.GaussianMixture(n_components=n_comp)
        GM.fit(Data)
        mixture = GM.weights_
        Mean = GM.means_
        Sigma = GM.covariances_
        
    if law_to_fit == "Mixture_Gaussian_Multivariate_sklearn":
        print("Fitting on a Mixture of Gaussian Multivariate law")
        GM = sklearn.mixture.GaussianMixture(n_components=n_comp)
        GM.fit(Data)
        mixture = GM.weights_
        Mean = GM.means_
        Sigma = GM.covariances_
        
    if law_to_fit == "Mixture_Gaussian_Multivariate_hand":
        print("Fitting on a Mixture Gaussian Multivariate law")
        params = np.array([data.get_hidden_parameter() for data in data_train])
        sorted = np.argsort(params)
        params_sorted = params[sorted]
        Data_sorted = np.array(Data)[sorted]
        
        bins = [int(len(params)/n_comp*count) for count in range(0,n_comp+1)]
        Mean = []  
        Sigma = [] 
        mixture = []
        for comp in range(0,n_comp):
            Mean.append(np.mean(Data_sorted[bins[comp]:bins[comp+1]], axis=0))
            Sigma.append(np.cov(Data_sorted[bins[comp]:bins[comp+1]], rowvar=0))
            mixture.append((len(params))/n_comp)
        #mixture = [mix/sum(mixture) for mix in mixture]
    return Mean,Sigma,mixture