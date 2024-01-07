from MLP_numpy import MLP

perc = MLP()

# print([arr.shape for arr in perc.weights])
# print([arr.shape for arr in perc.biases])

print(perc.forward().shape)
