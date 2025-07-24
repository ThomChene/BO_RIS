import numpy as np

#######################################################

class Signal_function:
    def __init__(self, number_antennas):
        self.number_antennas = number_antennas
    def generate(self)->np.ndarray:
        return np.ones(self.number_antennas)