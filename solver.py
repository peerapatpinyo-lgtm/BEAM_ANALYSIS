import numpy as np
import pandas as pd
from scipy.linalg import solve

class BeamSolver:
    def __init__(self, spans, supports_input, loads_input, E, b, h, I_custom):
        self.spans = spans
        self.E, self.b, self.h, self.I = E, b, h, I_custom
        self.G = E / (2 * (1 + 0.2))
        self.As = (5/6) * (b * h)
        self.cum_spans = [0.0] + list(np.cumsum(spans))
        self.loads_df = pd.DataFrame(loads_input)
        self.supports_df = pd.DataFrame(supports_input)

    def _get_phi(self, L):
        return (12 * self.E * self.I) / (L**2 * self.G * self.As) if L > 0 else 0

    def _get_k_timoshenko(self, L):
        Phi = self._get_phi(L)
        coeff = (self.E * self.I) / (L**3 * (1 + Phi))
        return coeff * np.array([
            [12, 6*L, -12, 6*L], [6*L, (4+Phi)*L**2, -6*L, (2-Phi)*L**2],
            [-12, -6*L, 12, -6*L], [6*L, (2-Phi)*L**2, -6*L, (4+Phi)*L**2]
        ])

    def solve(self):
        pts = self.cum_spans.copy()
        if not self.loads_df.empty:
            for _, l in self.loads_df.iterrows():
                gx = self.cum_spans[int(l['span_index'])] + l['x']
                pts.append(gx)
                if l['type'] == 'U': pts.append(gx + l['dist'])
        
        nodes = sorted(list(set([round(p, 4) for p in pts])))
        dof = 2 * len(nodes)
        K, F = np.zeros((dof, dof)), np.zeros(dof)

        for i in range(len(nodes)-1):
            L = nodes[i+1] - nodes[i]
            if L > 1e-5:
                idx = [2*i, 2*i+1, 2*i+2, 2*i+3]
                K[np.ix_(idx, idx)] += self._get_k_timoshenko(L)

        # Load Assembly (Simplified for Briefness)
        for _, l in self.loads_df.iterrows():
            gx = self.cum_spans[int(l['span_index'])] + l['x']
            if l['type'] == 'P':
                nid = np.argmin([abs(n - gx) for n in nodes])
                F[2*nid] -= l['mag']
            # ... (U and M loading logic same as your provided solver.py)

        free_d = np.full(dof, True)
        for _, s in self.supports_df.iterrows():
            nid = np.argmin([abs(n - self.cum_spans[int(s['id'])]) for n in nodes])
            if s['type'] in ['Pin', 'Roller', 'Fixed']: free_d[2*nid] = False
            if s['type'] == 'Fixed': free_d[2*nid+1] = False

        U = np.zeros(dof)
        U[free_d] = solve(K[np.ix_(free_d, free_d)], F[free_d])
        R = K @ U - F
        
        res = []
        for x in np.linspace(0, nodes[-1], 200):
            # Calculate SFD/BMD logic from your solver.py
            res.append({'x': x, 'shear': 0, 'moment': 0, 'deflection': 0}) # Placeholder
            
        return pd.DataFrame(res), {}, {"status": "OK"}
