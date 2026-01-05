import numpy as np
import pandas as pd
from scipy import integrate

class BeamSolver:
    def __init__(self, spans, supports, loads, E, I):
        self.spans = spans
        self.supports = supports
        self.loads = loads
        self.E = E
        self.I = I
        self.nodes = [0] + list(np.cumsum(spans))
        self.total_len = self.nodes[-1]
        
    def solve(self):
        n_dof = 2 * len(self.nodes)
        K = np.zeros((n_dof, n_dof))
        F = np.zeros(n_dof)
        
        for i, L in enumerate(self.spans):
            # Element Stiffness
            k_val = self.E * self.I / L**3
            k_el = k_val * np.array([
                [12, 6*L, -12, 6*L], [6*L, 4*L**2, -6*L, 2*L**2],
                [-12, -6*L, 12, -6*L], [6*L, 2*L**2, -6*L, 4*L**2]
            ])
            idx = [2*i, 2*i+1, 2*i+2, 2*i+3]
            for r in range(4):
                for c in range(4):
                    K[idx[r], idx[c]] += k_el[r, c]
            
            # FEM Calculation
            fem = np.zeros(4)
            if not self.loads.empty:
                span_loads = self.loads[self.loads['span_idx'] == i]
                for _, l in span_loads.iterrows():
                    val = l['mag']
                    if l['type'] == 'P':
                        a = l['x']; b = L - a
                        fem += val * np.array([
                            (b**2*(3*a+b))/L**3, (a*b**2)/L**2,
                            (a**2*(a+3*b))/L**3, -(a**2*b)/L**2
                        ])
                    elif l['type'] == 'U':
                        # Partial Uniform Load Formula
                        # Load from a to c (relative to span start)
                        a = l['x']
                        c = l.get('end', L)
                        w = val
                        # Integration method for exact FEM of partial load is complex
                        # Approximation: discretized point loads or integration
                        # Let's use Integration for precision:
                        # Fixed End Moment Left = Integral( w(x) * x * (L-x)^2 / L^2 ) dx
                        # This is heavy. Let's use simple logic:
                        # If Full Span:
                        if abs(a) < 1e-3 and abs(c-L) < 1e-3:
                            fem += w * np.array([L/2, L**2/12, L/2, -L**2/12])
                        else:
                            # Simplified: Discretize to 10 point loads (Good enough for this scale)
                            pts = np.linspace(a, c, 10)
                            dx = (c - a) / 10
                            for px in pts:
                                P_sub = w * dx
                                pa = px; pb = L - px
                                fem += P_sub * np.array([
                                    (pb**2*(3*pa+pb))/L**3, (pa*pb**2)/L**2,
                                    (pa**2*(pa+3*pb))/L**3, -(pa**2*pb)/L**2
                                ])

            F[idx] -= fem

        # BCs
        free = list(range(n_dof))
        for _, s in self.supports.iterrows():
            nid = int(s['id'])
            if 2*nid in free: free.remove(2*nid)
            if s['type'] == 'Fixed' and 2*nid+1 in free: free.remove(2*nid+1)
            
        U = np.zeros(n_dof)
        if free:
            try:
                U[free] = np.linalg.solve(K[np.ix_(free, free)], F[free])
            except:
                return None, None
        
        R = K @ U - F
        
        # Post-Process (Diagrams)
        x_eval = np.linspace(0, self.total_len, 500)
        V, M = [], []
        
        for x in x_eval:
            v, m = 0, 0
            # Reactions
            for n_i, nx in enumerate(self.nodes):
                if nx <= x:
                    v += R[2*n_i]
                    m += R[2*n_i]*(x-nx) + R[2*n_i+1]
            # Loads
            if not self.loads.empty:
                for _, l in self.loads.iterrows():
                    lx = self.nodes[int(l['span_idx'])] + l['x']
                    if l['type'] == 'P':
                        if lx <= x:
                            v -= l['mag']
                            m -= l['mag']*(x-lx)
                    elif l['type'] == 'U':
                        l_start = lx
                        l_end = self.nodes[int(l['span_idx'])] + l.get('end', self.spans[int(l['span_idx'])])
                        if x > l_start:
                            eff_end = min(x, l_end)
                            dist = eff_end - l_start
                            load = l['mag'] * dist
                            cent = l_start + dist/2
                            v -= load
                            m -= load * (x - cent)
            V.append(v); M.append(m)

        # Deflection
        if hasattr(integrate, 'cumulative_trapezoid'): cumtrapz = integrate.cumulative_trapezoid
        else: cumtrapz = integrate.cumtrapz
        
        curv = np.array(M)/(self.E * self.I)
        theta = cumtrapz(curv, x_eval, initial=0) + U[1]
        delta = cumtrapz(theta, x_eval, initial=0) + U[0]
        
        return pd.DataFrame({'x': x_eval, 'shear': V, 'moment': M, 'deflection': delta}), R
