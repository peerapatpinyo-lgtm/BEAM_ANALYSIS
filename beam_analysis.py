import numpy as np
import pandas as pd
from scipy import linalg

class BeamAnalysisEngine:
    def __init__(self, spans, supports, loads):
        self.spans = spans
        self.supports = supports
        self.loads = loads
        self.n_nodes = len(spans) + 1
        self.cum_len = [0] + list(np.cumsum(spans))

    def solve(self):
        # *Note: This simplified solver assumes uniform EI for force distribution.*
        
        # 1. Mesh
        x_eval = []
        for i, L in enumerate(self.spans):
            x_start = self.cum_len[i]
            pts = [0, L]
            for l in self.loads:
                if l['span_idx'] == i and l['type'] == 'P': pts.append(l['x'])
            pts = sorted(list(set(pts)))
            for j in range(len(pts)-1):
                seg = np.linspace(pts[j], pts[j+1], 25) # Increased resolution
                if j > 0: seg = seg[1:]
                x_eval.extend(x_start + seg)
        x_eval = np.array(sorted(list(set(x_eval))))
        
        # 2. Stiffness Setup
        NDOF = 2 * self.n_nodes
        K = np.zeros((NDOF, NDOF))
        F = np.zeros(NDOF)
        E, I = 2e10, 1e-4 # Dummy values
        
        for i, L in enumerate(self.spans):
            idx = 2*i
            k_el = (E*I/L**3) * np.array([[12, 6*L, -12, 6*L], [6*L, 4*L**2, -6*L, 2*L**2], [-12, -6*L, 12, -6*L], [6*L, 2*L**2, -6*L, 4*L**2]])
            K[idx:idx+4, idx:idx+4] += k_el
            
            # FEM
            fem = np.zeros(4)
            for l in [load for load in self.loads if load['span_idx']==i]:
                if l['type']=='U': w=l['w']; fem+=[w*L/2, w*L**2/12, w*L/2, -w*L**2/12]
                elif l['type']=='P': P=l['P']; a=l['x']; b=L-a; fem+=[P*b**2*(3*a+b)/L**3, P*a*b**2/L**2, P*a**2*(a+3*b)/L**3, -P*a**2*b/L**2]
            F[idx:idx+4] += fem

        # 3. BCs & Solve
        fixed = []
        for i, row in self.supports.iterrows():
            if row['type'] in ['Pin', 'Roller']: fixed.append(2*i)
            elif row['type'] == 'Fixed': fixed.extend([2*i, 2*i+1])
        active = [d for d in range(NDOF) if d not in fixed]
        if not active: return None, None

        try:
            D_a = linalg.solve(K[np.ix_(active, active)], -F[active])
            D = np.zeros(NDOF); D[active] = D_a
            R = K @ D + F
        except linalg.LinAlgError: return None, None
            
        # 4. Post-Process
        shear, moment = [], []
        for x in x_eval:
            V, M = 0, 0
            for i in range(self.n_nodes): # Reactions
                xn = self.cum_len[i]
                if xn < x: V += R[2*i]; M += R[2*i]*(x-xn) - R[2*i+1]
            for l in self.loads: # Loads
                xs = self.cum_len[l['span_idx']]
                if l['type']=='U':
                    xe = min(x, xs+self.spans[l['span_idx']])
                    if x>xs: len_a=xe-xs; V-=l['w']*len_a; M-=l['w']*len_a*(x-(xs+len_a/2))
                elif l['type']=='P':
                    xl = xs+l['x']
                    if x>xl: V-=l['P']; M-=l['P']*(x-xl)
            shear.append(V); moment.append(M)
            
        return pd.DataFrame({'x': x_eval, 'shear': shear, 'moment': moment}), R
