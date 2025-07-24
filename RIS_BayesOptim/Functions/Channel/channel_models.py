import numpy as np

#######################################################

class Channel_Model:
    """Parent Class for different channel models"""
    def __init__(self, dimension_matrix:tuple[int,int]):
        self.dimension_matrix = dimension_matrix
        
    def create_channel(self)->tuple[np.ndarray,list[float]]:
        return np.zeros(self.dimension_matrix),[0]
    
#######################################################

class geometric_far_field_channel_model(Channel_Model):
    """Class to generate a channel matrix with a geometric channel model. The antennas are equally spaced, in far field, and the antenna array is one dimensional:
    Dimension of the Matrix: dimension_matrix
    Number of paths (including LOS): number_paths
    Power of the LOS component over the sum of the powers of the NLOS: ratio_LOS_NLOS
    We fix the sum of power of all components (LOS,NLOS) to 1
    Each power of component NLOS is uniformly distributed, and they sum to 1/(self.ratio_LOS_NLOS + 1)
    Each attenuation is random Gaussian(mu,sigma) with mu^2+sigma^2 = power component
    ratio_sigma determines: mu^2 = ratio_sigma*power component
    Range of the angle of each path: range_angles"""
    
    def __init__(self, dimension_matrix:tuple[int,int],number_paths:int,ratio_LOS_NLOS:float,range_angles:list[tuple[tuple[float,float],tuple[float,float]]],ratio_sigma = 0.1):
        Channel_Model.__init__(self,dimension_matrix)
        self.number_paths = number_paths
        self.ratio_LOS_NLOS = ratio_LOS_NLOS
        self.range_angles = range_angles
        self.ratio_sigma = ratio_sigma
        
    def create_channel(self)->tuple[np.ndarray,list[float]]: 
        H = np.zeros((self.dimension_matrix[0],self.dimension_matrix[1]))
        lambda_paths = []
        if self.number_paths == 1:
            lambda_paths.append(1.0)
        else:
            lambda_paths.append(1-1/(self.ratio_LOS_NLOS + 1))
            c = [np.random.uniform(0,1) for i in range(1,self.number_paths)]
            S = sum(c)*(self.ratio_LOS_NLOS + 1)
            c = [c_i/S for c_i in c]
            for c_i in c:
                lambda_paths.append(c_i)
        
        for l in range(0,self.number_paths):
            theta_1 = np.random.uniform(self.range_angles[l][0][0],self.range_angles[l][0][1])
            theta_2 = np.random.uniform(self.range_angles[l][1][0],self.range_angles[l][1][1])
            if l == 0:
                theta_1_hidden,theta_2_hidden = -np.sin(theta_1),np.sin(theta_2)
            #att = np.random.normal(1,np.sqrt(lambda_paths[l]/2)) + 1j*np.random.normal(1,np.sqrt(lambda_paths[l]/2))
            #att = np.random.normal(1,np.sqrt(lambda_paths[l]))
            att = np.random.normal(np.sqrt(lambda_paths[l]/(1+self.ratio_sigma)),np.sqrt(lambda_paths[l]*(1+self.ratio_sigma)*self.ratio_sigma))
            
            h_1 = np.array([[np.exp(-1j * np.pi * index * np.sin(theta_1)) for index in range(0,self.dimension_matrix[0])]])/np.sqrt(self.dimension_matrix[0])
            h_2 = np.array([[np.exp(1j * np.pi * index * np.sin(theta_2)) for index in range(0,self.dimension_matrix[1])]])/np.sqrt(self.dimension_matrix[1])
            H = H + att*np.dot(h_1.T,h_2)  
        return H,[theta_1_hidden,theta_2_hidden]
    
#######################################################