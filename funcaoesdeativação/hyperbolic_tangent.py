import numpy as np

def hyperbolic_tangent(soma):
    return (np.exp(soma) - np.exp(-soma)) / (np.exp(soma) + np.exp(-soma))

print (hyperbolic_tangent(-10))
print (hyperbolic_tangent(10))
print (hyperbolic_tangent(1))
print (hyperbolic_tangent(0))