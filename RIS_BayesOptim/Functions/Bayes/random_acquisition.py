import numpy as np

from Functions.Channel import data_class

def random_acquisition(data_test:list[data_class.data_element],Number_uses_acquisition:int):
    """
    A random acquisition function, that test random points
    Returns: Average maximum value according to the number of points tested
    """
    average_ratio_at_each_query:np.ndarray = np.array([0] * (Number_uses_acquisition+1))
    for i in range(0,len(data_test)):
        data = data_test[i]
        ### First shuffle the indices to test ###
        shape_vector = data.get_shape_vector()
        Indices:np.ndarray = np.arange(0,shape_vector)
        np.random.shuffle(Indices)
        ### The points we receive ###
        All_points_received = []
        ratio_at_each_query = []
        
        current_argmax:int = Indices[0]
        ratio_at_each_query.append(data.get_ratio_element_value_dB(current_argmax))
        for query in range(0,Number_uses_acquisition):
            index_to_test = Indices[query] # Random index to test
            All_points_received.append(data.get_element_value_index(index_to_test,noise = True))  
            current_argmax = Indices[np.argmax(All_points_received)]
            ratio_at_each_query.append(data.get_ratio_element_value_dB(current_argmax))
        average_ratio_at_each_query = average_ratio_at_each_query + np.array(ratio_at_each_query) 
    return average_ratio_at_each_query/len(data_test)