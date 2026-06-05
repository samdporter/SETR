import pytest
import numpy as np

from tests._vtv_test_utils import require_vtv, torch
require_vtv()

from recon_core.core.gradients import Jacobian
from recon_core.priors.vtv.preconditioners import compute_precond_block
from recon_core.priors.vtv.vtv import WeightedVectorialTotalVariation

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_preconditioner_nan_handling():
    # Make a dummy 2-modality Jacobian
    nx, ny, nz = 4, 4, 4
    M = 2
    
    # We create x_arr with NaNs simulating a bad bed position or numerical instability
    x_arr = torch.rand((nx, ny, nz, M), dtype=torch.float32, device=DEVICE)
    x_arr[1, 1, 1, :] = float('nan')
    x_arr[2, 2, 2, :] = float('inf')
    
    class DummyJacobian:
        def __init__(self):
            self.grad = type('Grad', (), {'directions': [(1,0,0), (0,1,0), (0,0,1)], 'bnd_cond': 'Neumann'})()
            
        def direct(self, x):
            # returns gradients approx shape (nx, ny, nz, M, d=3)
            d = 3
            out = torch.rand((nx, ny, nz, M, d), dtype=torch.float32, device=DEVICE)
            out[1, 1, 1, :, :] = float('nan') # prop NaNs
            out[2, 2, 2, :, :] = float('inf') # prop Infs
            return out

        def sensitivity(self, x):
            d = 3
            out = torch.ones((nx, ny, nz, M, d), dtype=torch.float32, device=DEVICE)
            return out
    # We need a wvtv object with .jacobian, .smoothing, .vtv.eps
    class DummyWVTV:
        def __init__(self):
            self.jacobian = DummyJacobian()
            self.smoothing = "charbonnier"
            self.vtv = type('VTV', (), {'eps': 1e-4})()
            self.weights = torch.tensor([1.0, 1.0], dtype=torch.float32, device=DEVICE)
            
    wvtv = DummyWVTV()
    
    # this call should not raise CUSOLVER_STATUS_INVALID_VALUE 
    # and should return normally even if inputs are NaNs or Infs
    res = compute_precond_block(wvtv, x_arr, method="mm_diag_block_maj", epsilon=1e-8)
    
    assert res is not None
    # the nan voxels should come out as diagonal epsilon blocks or zeros
    # the method has _spd_floor_blocks applied at the end so it should be valid SPD floors
    assert not torch.isnan(res).any()
    assert not torch.isinf(res).any()
    
