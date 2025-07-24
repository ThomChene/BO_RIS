import numpy as np
import matplotlib.pyplot as plt
from Functions.Channel import data_class
import pandas as pd

def visualize_3D(data_train:list[data_class.data_element],coordinates:list[int],save:bool=True,filename:str="./results.dat"):
    frame = {}
    x = [data.get_element_value_index(coordinates[0]) for data in data_train]
    y = [data.get_element_value_index(coordinates[1]) for data in data_train]
    z = [data.get_element_value_index(coordinates[2]) for data in data_train]
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x,y,z)
    frame["X"] = x
    frame["Y"] = y
    frame["Z"] = z
    if save:
        output_df = pd.DataFrame(frame)
        output_df.to_csv(filename)
    
def visualize_hist(data_train:list[data_class.data_element],coordinate:int,save:bool=True,filename:str="./results.dat"):
    frame = {}
    x = [data.get_element_value_index(coordinate) for data in data_train]
    fig = plt.figure()
    counts, bins = np.histogram(x)
    counts = counts/len(x)
    plt.stairs(counts, bins)
    counts = np.append(counts,0) # To have the same size than bins
    frame["Counts"] = counts
    frame["Bins"] = bins
    if save:
        output_df = pd.DataFrame(frame)
        output_df.to_csv(filename)
    
def visualize_beam(data_train:list[data_class.data_element],coordinates:list,save:bool=True,filename:str="./results.dat"):
    frame = {}
    fig = plt.figure()
    for coord in coordinates:
        y = np.array([data.get_element_value_index(coord) for data in data_train])
        x = np.array([data.get_hidden_parameter() for data in data_train])
        sorted = np.argsort(x)
        y = y[sorted]
        x = x[sorted]
        plt.scatter(x,y,label = coord)
        frame["x_{0}".format(coord)] = x
        frame["y_{0}".format(coord)] = y
    plt.legend()
    if save:
        output_df = pd.DataFrame(frame)
        output_df.to_csv(filename)
    
def plot_results(*results:tuple[np.ndarray,str],save:bool=True,filename:str="./results.dat"):
    fig = plt.figure()
    frame = {}
    max_length = max([len(acquisition[0]) for acquisition in results])
    for acquisition in results:
        name = acquisition[1]
        values = acquisition[0]
        for fill in range(len(values),max_length):
            values = np.append(values,values[-1])
        frame[name] = values
        plt.plot(np.arange(0,len(values)),values,label=name)
    plt.plot(np.arange(0,len(values)),np.zeros(len(values)))
    plt.legend()
    if save:
        output_df = pd.DataFrame(frame)
        output_df.to_csv(filename)