import numpy as np

def softmax(soma):
    ex = np.exp(soma)
    return ex / ex.sum()

print(softmax([2.0, 1.0, 0.1]))
print(softmax([10, 9, 8]))
print(softmax([1, 2, 3]))
print(softmax([0, 0, 0]))
# Output:
