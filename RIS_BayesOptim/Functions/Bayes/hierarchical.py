import numpy as np

from Functions.Channel import data_class

def hierarchical_acquisition(data_test:list[data_class.data_element],Number_uses_acquisition:int,len_tree:int):
    """
    Hierarchical method
    The order of the beams should be: widest,narrower,...,narrowest
    Binary Search
    Returns: Average maximum value according to the number of points tested
    """
    average_ratio_at_each_query:np.ndarray = np.array([0] * (2*len_tree+1))
    for i in range(0,len(data_test)):
        data = data_test[i]
        ### The points we receive ###
        All_points_received = []
        All_indices_received = []
        ratio_at_each_query = []
        
        current_argmax = 0
        ratio_at_each_query.append(data.get_ratio_element_value_dB(current_argmax))
        last_best:int = 0
        
        ## Test two narrowest beams
        beam_1 = data.get_element_value_index(0,noise = True)
        All_points_received.append(data.get_element_value_index(0,noise = True))
        All_indices_received.append(0)
        current_argmax = All_indices_received[np.argmax(All_points_received)]
        ratio_at_each_query.append(data.get_ratio_element_value_dB(current_argmax))
        
        beam_2 = data.get_element_value_index(1,noise = True)
        All_points_received.append(data.get_element_value_index(1,noise = True))
        All_indices_received.append(1)
        current_argmax = All_indices_received[np.argmax(All_points_received)]
        ratio_at_each_query.append(data.get_ratio_element_value_dB(current_argmax))
        
        if beam_1>beam_2:
            last_best = 0
        else:
            last_best = 1
        for query in range(0,2*len_tree-2):
            if query%2 == 0:
                index_to_test = 0 + 2*last_best + 2
                beam_1 = data.get_element_value_index(index_to_test,noise = True)
                All_points_received.append(data.get_element_value_index(index_to_test,noise = True))
            if query%2 == 1:
                index_to_test = 1 + 2*last_best + 2
                beam_2 = data.get_element_value_index(index_to_test,noise = True)
                All_points_received.append(data.get_element_value_index(index_to_test,noise = True))
                if beam_1>beam_2:
                    last_best = 2*last_best + 2
                else:
                    last_best = 1 + 2*last_best + 2
                  
            All_indices_received.append(index_to_test)
            current_argmax = All_indices_received[np.argmax(All_points_received)]
            ratio_at_each_query.append(data.get_ratio_element_value_dB(current_argmax))
        average_ratio_at_each_query = average_ratio_at_each_query + np.array(ratio_at_each_query) 
    return average_ratio_at_each_query/len(data_test)