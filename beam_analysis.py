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
        # 1. Mesh Creation
        x_eval = []
        for i, L in enumerate(self.spans):
            x_start = self.cum_len[i]
            pts = [0, L]
            for l in self.loads:
                if l['span_idx'] == i and l['type'] == 'P':
                    pts.append(l['x'])
            pts = sorted(list(set(pts)))
            
            for j in range(len(pts)-1):
                seg = np.linspace(pts[j], pts[j+1], 20)
                if j > 0: seg = seg[1:]
                x_eval.extend(x_start + seg)
        
        x_eval = np.array(sorted(list(set(x_eval))))
        
        # 2. Stiffness Matrix
        NDOF = 2 * self.n_nodes
        K_global = np.zeros((NDOF, NDOF))
        F_global = np.zeros(NDOF)
        
        E = 2e10 
        I = 1e-4 
        
        for i, L in enumerate(self.spans):
            idx1, idx2 = 2*i, 2*(i+1)
            k = (E*I/L**3) * np.array([
                [12, 6*L, -12, 6*L],
                [6*L, 4*L**2, -6*L, 2*L**2],
                [-12, -6*L, 12, -6*L],
                [6*L, 2*L**2, -6*L, 4*L**2]
            ])
            K_global[idx1:idx1+4, idx1:idx1+4] += k
            
            fem = np.zeros(4)
            span_loads = [l for l in self.loads if l['span_idx'] == i]
            for l in span_loads:
                if l['type'] == 'U':
                    w = l['w']; fem[0]+=w*L/2; fem[2]+=w*L/2; fem[1]+=w*L**2/12; fem[3]-=w*L**2/12
                elif l['type'] == 'P':
                    P=l['P']; a=l['x']; b=L-a
                    fem[0]+=P*b**2*(3*a+b)/L**3; fem[2]+=P*a**2*(a+3*b)/L**3
                    fem[1]+=P*a*b**2/L**2; fem[3]-=P*a**2*b/L**2
            F_global[idx1:idx1+4] += fem

        # 3. Boundary Conditions
        fixed_dofs = []
        for i, row in self.supports.iterrows():
            stype = row['type']
            if stype in ['Pin', 'Roller']: fixed_dofs.append(2*i)
            elif stype == 'Fixed': fixed_dofs.extend([2*i, 2*i+1])
            
        active_dofs = [d for d in range(NDOF) if d not in fixed_dofs]
        
        if not active_dofs: return None, None

        K_aa = K_global[np.ix_(active_dofs, active_dofs)]
        F_a = -F_global[active_dofs]
        
        try:
            D_a = linalg.solve(K_aa, F_a)
        except linalg.LinAlgError:
            return None, None
            
        D_total = np.zeros(NDOF)
        D_total[active_dofs] = D_a
        R_total = K_global @ D_total + F_global
        
        # 4. Post-Processing
        shear, moment = [], []
        for x in x_eval:
            V, M = 0, 0
            for n_i in range(self.n_nodes):
                xn = self.cum_len[n_i]
                if xn < x:
                    Ry, Rm = R_total[2*n_i], R_total[2*n_i+1]
                    dist = x - xn
                    if dist > 1e-5:
                        V += Ry
                        M += Ry*dist - Rm
            for l in self.loads:
                xs = self.cum_len[l['span_idx']]
                if l['type'] == 'U':
                    xe = xs + self.spans[l['span_idx']]
                    if x > xs:
                        len_act = min(x, xe) - xs
                        load = l['w'] * len_act
                        cent = x - (xs + len_act/2)
                        V -= load
                        M -= load * cent
                elif l['type'] == 'P':
                    xl = xs + l['x']
                    if x > xl:
                        V -= l['P']
                        M -= l['P'] * (x - xl)
            shear.append(V)
            moment.append(M)
            
        return pd.DataFrame({'x': x_eval, 'shear': shear, 'moment': moment}), R_total
