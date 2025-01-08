import numpy as np

def relu(soma):
    return np.maximum(0, soma) # retorna o maior valor entre 0 e soma 

print(relu(2.1))
print(relu(10))
print(relu(1))
print(relu(0))
# Output: