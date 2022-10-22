from perceptron.perceptron import Perceptron

neuron = Perceptron(2)

"""Implementation of the NOR gate."""
neuron.set_weights([-20,-20,15]) 

print("Gate:")
print ("0 0 = {0:.10f}".format(neuron.run([0,0])))
print ("0 1 = {0:.10f}".format(neuron.run([0,1])))
print ("1 0 = {0:.10f}".format(neuron.run([1,0])))
print ("1 1 = {0:.10f}".format(neuron.run([1,1])))
