import numpy as np

#sigmoid function
def nonlin(x,deriv=False):
    if deriv == True:
        return x * (1-x)

    return 1/(1+np.exp(-x))


# input data
# 4 training examples with 3 input neurons each
X = np.array([[0,0,1],
              [0,1,1],
              [1,0,1],
              [1,1,1]])

# ouput data
# 1 output neuron for each training example
Y = np.array([[0],
              [1],
              [1],
              [0]])

# passing same seed for every execution of program for the purpose of debugging
np.random.seed(1)


# This network has 3 network layers
# therefore it needs two synapse matrices
# Synapse is connections between two network layers
syn0 = 2 * np.random.random((3,4)) - 1
syn1 = 2 * np.random.random((4,1)) - 1


#training step
for j in range(60000):

    #first layer = input data
    l0 = X
    # matrix multiplication between each layer and synpase
    l1 = nonlin(np.dot(l0, syn0))
    l2 = nonlin(np.dot(l1, syn1))

    # finding error value
    l2_Error = Y - l2

    # average error rate at a set interval to maek sure it goes down everytime
    if (j % 10000) == 0:
        print("Error "+ str(np.mean(np.abs(l2_Error))))

    # derivative of the output from layer 2
    # this delta is then used to update the synapses every iteration
    l2_delta = l2_Error*nonlin(l2, deriv=True)

    # find out how much layer 1 contributed to error in layer2
    # this is called backpropogation
    # this is calculated as layer2's delta * transpose of synapse 1
    l1_Error = l2_delta.dot(syn1.T)

    l1_delta = l1_Error*nonlin(l1, deriv=True)

    #use delta from every layer to update synapse weights to reduce error rate every iteration
    # gradient descent algorithm
    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)


#print output
print("printing output")
print(l2)
