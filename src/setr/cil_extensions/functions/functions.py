import numpy as np
from cil.optimisation.functions import Function

class BlockIndicatorBox(Function):
    
    def __init__(self, lower = 0, upper = np.inf):
        self.lower = lower
        self.upper = upper
        
    def __call__(self, x):
        # because we're using this as a projection, this should always return 0
        # se we'll be a bit cheeky and return 0.0 to save computation time
        #TODO: change this to work as an actual indicator function
        return 0.0

    def proximal(self, x, tau, out=None):
        if out is None:
            out = x.copy()
        out.fill(x.maximum(self.lower))
        out.fill(out.minimum(self.upper))
        return out 