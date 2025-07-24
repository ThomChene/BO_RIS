import numpy as np

class data_element:
    """In the Dataset, every element is a vector:
    Type feedback: 
    Gaussian: The output is a gaussian
    Quadratic: The output is quadratic (x.H A x), always positive
    dB: The output is in dB: 10*log10()
    
    """
    def __init__(self, element:np.ndarray,hidden:float=0):
        self.element = element
        self.length = len(element)
        self.hidden = hidden
        
    def add_hidden_parameter(self,hidden:float):
        self.hidden = hidden
        
    def get_hidden_parameter(self):
        return self.hidden
    
    def get_element_value_index(self,index:int,noise:bool=False)->float:
        if noise:
            if self.type_noise == "Gaussian":
                w = np.random.normal(self.mean_noise,self.var_noise)
                return self.element[index] + w
        return self.element[index]
    
    def get_ratio_element_value_dB(self,index:int)->float:
        value = self.get_element_value_index(index,noise=False)
        max,arg = self.get_real_max()
        
        if self.type_feedback == "Gaussian":
            return value - max
        
        elif self.type_feedback == "Quadratic":
            return 10*np.log10(value/(max+1e-5))
        
        elif self.type_feedback == "dB":
            return value - max
        
        else:
            return value - max
        
    def set_feedback_model(self,type_noise:str="Gaussian",mean_noise:float=0,var_noise:float = 0,type_feedback:str = "Quadratic"):
        self.type_noise = type_noise
        self.mean_noise = mean_noise
        self.var_noise = var_noise
        self.type_feedback = type_feedback
        
        if type_feedback == "Gaussian":
            vector = self.get_element()
            self.set_element(vector)
            
        if type_feedback == "Quadratic":
            vector = self.get_element()
            self.set_element(vector)
            
        if type_feedback == "dB":
            vector = self.get_element()
            self.set_element(10*np.log10(vector+1e-5))
            
        self.set_real_max()
        
    def get_shape_vector(self):
        return self.length 
    
    def get_element(self):
        return self.element
    
    def set_element(self,element):
        self.element = element
    
    def set_real_max(self):
        if self.type_noise == "Gaussian":
            self.real_max = max(self.element) + self.mean_noise
        self.argmax = np.argmax(self.element)
    
    def get_real_max(self)->tuple[float,int]:
        return self.real_max,self.argmax