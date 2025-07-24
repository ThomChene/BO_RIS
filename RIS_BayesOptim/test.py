import numpy as np
import matplotlib.pyplot as plt

dimension_data_parameters = 10
size_data_train = 10000

Mean = np.ones(dimension_data_parameters)
#Mean = np.random.rand(number_indices)
Sigma = np.array(1/2-np.random.rand(dimension_data_parameters,dimension_data_parameters))
Sigma = np.dot(Sigma.T,Sigma) # The covariance needs to be symmetric
data_train_numpy = np.random.multivariate_normal(Mean,Sigma,size_data_train)

Argmax = [np.argmax(data_train_numpy[i]) for i in range(0,size_data_train)]

a = np.histogram(Argmax)

plt.plot(np.arange(0,10),a[0]/size_data_train)
plt.plot(np.diag(Sigma))
plt.show()

print("a{0}".format(1))

