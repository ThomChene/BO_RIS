import numpy as np

#######################################################

class Codebook:
    """Parent Class for different codebooks"""
    def __init__(self, dimension_channel:int):
        self.dimension_channel = dimension_channel
        self.codebook:list = []
        
    def add_codeword(self,codeword:np.ndarray):
        self.codebook.append(codeword)
    
    def get_codewords(self)->list:
        return self.codebook
    
    def get_dimension_channel(self)->int:
        return self.dimension_channel
    
    def fuse_codebooks(self,other_codebook):
        if other_codebook.get_dimension_channel() == self.dimension_channel:
            other_codewords = other_codebook.get_codewords()
            for codeword in other_codewords:
                self.add_codeword(codeword)
                
    def add_constraints(self,phase_shifts:np.ndarray,amplitude:np.ndarray,constraint:str,n_bits:int=1)->np.ndarray:
        if constraint == 'None':
            codeword_list = [amplitude[n]*np.exp(1j*phase_shifts[n]) for n in range(0,self.dimension_channel)]
            codeword = np.array(codeword_list)
            
        if constraint == 'Power':
            codeword_list = [amplitude[n]*np.exp(1j*phase_shifts[n]) for n in range(0,self.dimension_channel)]
            codeword = np.array(codeword_list)/np.sqrt(sum(codeword_list**2))
            
        if constraint == 'Quantized':
            quantized = [2*np.pi*n/2**n_bits for n in range(2**n_bits)]
            value_phases = [(phase_shifts[n])%(2*np.pi) for n in range(0,self.dimension_channel)]
            codeword_list = [amplitude[n]*np.exp(1j*quantized[np.argmin(np.abs(value_phases[n]-quantized))]) for n in range(0,self.dimension_channel)]
            #print([amplitude[n]*np.exp(1j*phase_shifts[n]) for n in range(0,self.dimension_channel)])
            #print(quantized)
            codeword = np.array(codeword_list)  
        return codeword

#######################################################
  
class Narrow_codebook(Codebook):
    """Create a narrow beam codebook"""
    def __init__(self, dimension_channel:int,number_narrow_beams:int,range_beams:tuple[float,float],constraint:str,n_bits:int=1):
        Codebook.__init__(self,dimension_channel)
        self.number_narrow_beams = number_narrow_beams
        self.range_beams = range_beams
        self.constraint = constraint
        self.n_bits = n_bits
        self.create_codebook()
        
    def create_codebook(self):
        range_in_sin_0 = self.range_beams[0]
        range_in_sin_1 = self.range_beams[1]
        I = [range_in_sin_0 + index*(range_in_sin_1-range_in_sin_0)/self.number_narrow_beams + (range_in_sin_1-range_in_sin_0)/self.number_narrow_beams/2 for index in range(0,self.number_narrow_beams)]
        #I = [self.range_beams[0] + index*(self.range_beams[1]-self.range_beams[0])/self.number_narrow_beams + (self.range_beams[1]-self.range_beams[0])/self.number_narrow_beams/2 for index in range(0,self.number_narrow_beams)]
        for beam in range(0,self.number_narrow_beams):
            angle = I[beam]
            phase_shifts = np.array([angle*n for n in range(0,self.dimension_channel)])
            amplitude = np.array([1 for n in range(0,self.dimension_channel)])
            codeword = self.add_constraints(phase_shifts,amplitude,self.constraint,self.n_bits)
            self.add_codeword(codeword)
        
#######################################################

class Wide_codebook(Codebook):
    """Create a wide beam codebook
    Width factor: Each wide beam will have a size range beams/width_factor """
    def __init__(self, dimension_channel:int,number_wide_beams:int,range_beams:tuple[float,float],width_factor:float,constraint:str,n_bits:int=1):
        Codebook.__init__(self,dimension_channel)
        self.number_wide_beams = number_wide_beams
        self.width_factor = width_factor 
        self.range_beams = range_beams
        self.constraint = constraint
        self.n_bits = n_bits
        self.create_codebook()
        
    def create_codebook(self):
        assert self.width_factor < self.dimension_channel
        range_in_sin_0 = self.range_beams[0]
        range_in_sin_1 = self.range_beams[1]
        I = [range_in_sin_0 + index*(range_in_sin_1-range_in_sin_0)/self.number_wide_beams + (range_in_sin_1-range_in_sin_0)/self.number_wide_beams/2 for index in range(0,self.number_wide_beams)]
        for beam in range(0,self.number_wide_beams):
            angle = I[beam]
            phase_shifts = [angle*n for n in range(0,self.width_factor)]
            phase_shifts += [0]*(self.dimension_channel-self.width_factor)
            phase_shifts = np.array(phase_shifts)
            amplitude = [1 for n in range(0,self.width_factor)]
            amplitude += [0]*(self.dimension_channel-self.width_factor)
            amplitude = np.array(amplitude)
            
            codeword = self.add_constraints(phase_shifts,amplitude,self.constraint,self.n_bits)
            self.add_codeword(codeword)
       
#######################################################
  
class Random_codebook(Codebook):
    """Create a narrow beam codebook"""
    def __init__(self, dimension_channel:int,number_beams:int,constraint:str,n_bits:int=1):
        Codebook.__init__(self,dimension_channel)
        self.number_beams = number_beams
        self.constraint = constraint
        self.n_bits = n_bits
        self.create_codebook()
        
    def create_codebook(self):
        for beam in range(0,self.number_beams):
            phase_shifts = np.array([2*np.pi*np.random.rand() for n in range(0,self.dimension_channel)])
            amplitude = np.array([1 for n in range(0,self.dimension_channel)])
            codeword = self.add_constraints(phase_shifts,amplitude,self.constraint,self.n_bits)
            self.add_codeword(codeword)
             
#######################################################