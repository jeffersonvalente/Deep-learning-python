import numpy as np

def sigmoid(soma):
    return 1 / (1 + np.exp(-soma))

print(sigmoid(2.1))
print(sigmoid(10))
print(sigmoid(1))

  