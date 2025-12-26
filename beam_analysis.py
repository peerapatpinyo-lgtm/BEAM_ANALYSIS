import numpy as np
import pandas as pd

class SimpleBeamSolver:
    def __init__(self, spans, supports, loads):
        self.spans = spans
        self.supports = supports
        self.loads = loads

    def solve(self):
        # Mock calculation (สามารถเปลี่ยนเป็น Matrix Method ของจริงได้ในอนาคต)
        return True, None 

    def get_internal_forces(self, n_points=100):
        L = sum(self.spans)
        x = np.linspace(0, L, n_points)
        # Mockup กราฟแรงเฉือนและโมเมนต์
        shear = 15000 * np.cos(x * 1.5) * (1 - x/(L*1.2))
        moment = 30000 * np.sin(x * 1.5) * x/2
        return pd.DataFrame({'x': x, 'shear': shear, 'moment': moment})
