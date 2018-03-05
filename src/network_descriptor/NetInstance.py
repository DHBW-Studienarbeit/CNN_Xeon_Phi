from network_descriptor.network.Network import Network
from network_descriptor.activationshapes.StdActivationShape import StdActivationShape


input_shape = StdActivationShape(10, 28, 28, 1)
net = Network(input_shape)

net.add_convolutional_layer(5, 5, 60)
net.add_maxpooling_layer(2, 2)
net.add_convolutional_layer(5, 5, 80)
net.add_maxpooling_layer(2, 2)
net.add_fullyconnected_layer(10)

net.generate()
