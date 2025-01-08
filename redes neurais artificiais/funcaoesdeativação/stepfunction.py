import numpy as np

def step_function(soma):
    if soma >= 1:
        return 1
    return 0

print(step_function(0))
print(step_function(1))
print(step_function(-1))