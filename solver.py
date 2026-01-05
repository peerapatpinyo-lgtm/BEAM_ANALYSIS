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
        self.n_nodes = len(self.nodes)
        self.total_len = self.nodes[-1]
        
    def solve(self):
        # 1. Global Stiffness Matrix
        n_dof = 2 * self.n_nodes
        K = np.zeros((n_dof, n_dof))
        F = np.zeros(n_dof)
        
        for i, L in enumerate(self.spans):
            k = self.E * self.I / L**3
            k_el = k * np.array([
                [12, 6*L, -12, 6*L],
                [6*L, 4*L**2, -6*L, 2*L**2],
                [-12, -6*L, 12, -6*L],
                [6*L, 2*L**2, -6*L, 4*L**2]
            ])
            idx = [2*i, 2*i+1, 2*i+2, 2*i+3]
            for r in range(4):
                for c in range(4):
                    K[idx[r], idx[c]] += k_el[r, c]
                    
            # Fixed End Moments (FEM)
            fem = np.zeros(4)
            if not self.loads.empty:
                span_loads = self.loads[self.loads['span_idx'] == i]
                for _, l in span_loads.iterrows():
                    val = l['mag']
                    if l['type'] == 'P':
                        a = l['x']; b = L - a
                        fem += val * np.array([
                            (b**2 * (3*a+b))/L**3, (a * b**2)/L**2,
                            (a**2 * (a+3*b))/L**3, -(a**2 * b)/L**2
                        ])
                    elif l['type'] == 'U':
                        # Assuming full span uniform for simplicity
                        fem += val * np.array([L/2, L**2/12, L/2, -L**2/12])
            
            F[idx] -= fem

        # 2. Boundary Conditions
        free_dof = list(range(n_dof))
        for _, s in self.supports.iterrows():
            node = int(s['id'])
            if 2*node in free_dof: free_dof.remove(2*node) # Fix Y
            if s['type'] == 'Fixed' and (2*node+1 in free_dof):
                free_dof.remove(2*node+1) # Fix Rotation

        # 3. Solve
        U = np.zeros(n_dof)
        if free_dof:
            try:
                U[free_dof] = np.linalg.solve(K[np.ix_(free_dof, free_dof)], F[free_dof])
            except:
                return None, None
                
        # 4. Reactions
        R = K @ U - F
        
        # 5. Internal Forces (Integration Method)
        # Create dense points
        x_vals = sorted(list(set(
            list(np.linspace(0, self.total_len, 500)) + 
            self.nodes + 
            [n + 0.001 for n in self.nodes] + [n - 0.001 for n in self.nodes]
        )))
        # Add load points
        if not self.loads.empty:
            for _, l in self.loads.iterrows():
                lx = self.nodes[int(l['span_idx'])] + l['x']
                x_vals.extend([lx, lx-0.001, lx+0.001])
        x_vals = sorted(list(set([x for x in x_vals if 0 <= x <= self.total_len])))
        
        V_res, M_res = [], []
        
        for x in x_vals:
            v, m = 0, 0
            # Reactions contribution
            for i, node_x in enumerate(self.nodes):
                if node_x <= x + 1e-6:
                    v += R[2*i]
                    m += R[2*i] * (x - node_x) + R[2*i+1]
            
            # Loads contribution
            if not self.loads.empty:
                for _, l in self.loads.iterrows():
                    l_start = self.nodes[int(l['span_idx'])]
                    if l['type'] == 'P':
                        lp = l_start + l['x']
                        if lp <= x + 1e-6:
                            v -= l['mag']
                            m -= l['mag'] * (x - lp)
                    elif l['type'] == 'U':
                        l_end = self.nodes[int(l['span_idx']) + 1]
                        # Effective length of load to the left of x
                        eff_start = l_start
                        eff_end = min(x, l_end)
                        if eff_end > eff_start:
                            dist = eff_end - eff_start
                            load = l['mag'] * dist
                            centroid = eff_start + dist/2
                            v -= load
                            m -= load * (x - centroid)
                            
            V_res.append(v)
            M_res.append(m)
            
        # Deflection (Double Integration of M/EI)
        # Numerical integration
        if hasattr(integrate, 'cumulative_trapezoid'): cumtrapz = integrate.cumulative_trapezoid
        else: cumtrapz = integrate.cumtrapz

        curvature = np.array(M_res) / (self.E * self.I)
        slope = cumtrapz(curvature, x_vals, initial=0) + U[1] # Add initial slope
        defl = cumtrapz(slope, x_vals, initial=0) + U[0]      # Add initial defl
        
        return pd.DataFrame({'x': x_vals, 'shear': V_res, 'moment': M_res, 'deflection': defl}), R
