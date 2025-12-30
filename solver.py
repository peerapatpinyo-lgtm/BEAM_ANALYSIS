import numpy as np
import pandas as pd
from scipy import integrate

class BeamSolver:
    def __init__(self, spans, supports, loads, E=2e6, I=1e-4):
        self.spans = spans
        self.supports = supports 
        self.loads = loads
        self.E = E
        self.I = I
        self.nodes = [0] + list(np.cumsum(spans))
        self.n_nodes = len(self.nodes)
        
    def solve(self):
        n_dof = 2 * self.n_nodes 
        K_global = np.zeros((n_dof, n_dof))
        F_global = np.zeros(n_dof)
        
        # 1. Stiffness & Load Vector
        for i, L in enumerate(self.spans):
            k_val = (self.E * self.I / L**3)
            k_el = k_val * np.array([
                [12, 6*L, -12, 6*L], [6*L, 4*L**2, -6*L, 2*L**2],
                [-12, -6*L, 12, -6*L], [6*L, 2*L**2, -6*L, 4*L**2]
            ])
            
            idx = [2*i, 2*i+1, 2*i+2, 2*i+3]
            for r in range(4):
                for c in range(4):
                    K_global[idx[r], idx[c]] += k_el[r, c]
            
            fem = np.zeros(4)
            span_loads = [l for l in self.loads if int(l['span_idx']) == i]
            
            for load in span_loads:
                val = load['mag']
                if load['type'] == 'P':
                    a = load['x']; b = L - a; P = val
                    fem[0] += P * b**2 * (3*a + b) / L**3
                    fem[1] += P * a * b**2 / L**2
                    fem[2] += P * a**2 * (a + 3*b) / L**3
                    fem[3] -= P * a**2 * b / L**2
                elif load['type'] == 'U':
                    w = val
                    fem[0] += w * L / 2; fem[1] += w * L**2 / 12
                    fem[2] += w * L / 2; fem[3] -= w * L**2 / 12
                elif load['type'] == 'M':
                    M_app = val 
                    a = load['x']; b = L - a
                    # Correct FEM for Moment Load
                    fem[0] += -6 * M_app * a * b / L**3
                    fem[1] += M_app * b * (2*a - b) / L**2
                    fem[2] += 6 * M_app * a * b / L**3
                    fem[3] += M_app * a * (2*b - a) / L**2 # Fixed syntax 2*b
            
            F_global[idx] -= fem 

        # 2. Boundary Conditions
        free_dofs = list(range(n_dof))
        for _, row in self.supports.iterrows():
            node_idx = int(row['id'])
            if 2*node_idx in free_dofs: free_dofs.remove(2*node_idx) 
            if row['type'] == 'Fixed':
                if 2*node_idx+1 in free_dofs: free_dofs.remove(2*node_idx+1)

        # 3. Solve
        U_global = np.zeros(n_dof)
        if free_dofs:
            try:
                U_global[free_dofs] = np.linalg.solve(K_global[np.ix_(free_dofs, free_dofs)], F_global[free_dofs])
            except:
                return pd.DataFrame(), np.zeros(n_dof)

        # 4. Results
        results = []
        
        # --- FIX: Compatibility for SciPy ---
        if hasattr(integrate, 'cumulative_trapezoid'):
            trapz_func = integrate.cumulative_trapezoid
        else:
            trapz_func = integrate.cumtrapz
        # ------------------------------------

        for i, L in enumerate(self.spans):
            idx = [2*i, 2*i+1, 2*i+2, 2*i+3]
            u_el = U_global[idx]
            
            fem_local = np.zeros(4)
            span_loads = [l for l in self.loads if int(l['span_idx']) == i]
            
            eval_points = set(np.linspace(0, L, 100))
            for load in span_loads:
                eval_points.add(load['x'])
                if load['x'] > 0: eval_points.add(load['x'] - 1e-6)
                if load['x'] < L: eval_points.add(load['x'] + 1e-6)
                
                val = load['mag']
                if load['type'] == 'P':
                    a = load['x']; b = L - a
                    fem_local[0] += val*b**2*(3*a+b)/L**3; fem_local[1] += val*a*b**2/L**2
                    fem_local[2] += val*a**2*(a+3*b)/L**3; fem_local[3] -= val*a**2*b/L**2
                elif load['type'] == 'U':
                    fem_local[0] += val*L/2; fem_local[1] += val*L**2/12
                    fem_local[2] += val*L/2; fem_local[3] -= val*L**2/12
                elif load['type'] == 'M':
                    a = load['x']; b = L - a
                    fem_local[0] += -6*val*a*b/L**3; fem_local[1] += val*b*(2*a-b)/L**2
                    fem_local[2] += 6*val*a*b/L**3; fem_local[3] += val*a*(2*b-a)/L**2

            x_eval = sorted(list(eval_points))
            
            k_val = (self.E * self.I / L**3)
            k_el = k_val * np.array([
                [12, 6*L, -12, 6*L], [6*L, 4*L**2, -6*L, 2*L**2],
                [-12, -6*L, 12, -6*L], [6*L, 2*L**2, -6*L, 4*L**2]
            ])
            f_start = k_el @ u_el + fem_local
            V_start, M_start = f_start[0], f_start[1]
            
            M_vals, V_vals = [], []
            for x in x_eval:
                V_x = V_start
                M_x = -M_start + V_start * x 
                for load in span_loads:
                    lx = load['x']; mag = load['mag']
                    if load['type'] == 'P':
                        if x >= lx + 1e-9: V_x -= mag; M_x -= mag * (x - lx)
                    elif load['type'] == 'U':
                        if x > 0: d=x; V_x -= mag*d; M_x -= mag*d*d/2
                    elif load['type'] == 'M':
                        if x >= lx + 1e-9: M_x += mag 
                V_vals.append(V_x)
                M_vals.append(M_x)
                
            curv = np.array(M_vals) / (self.E * self.I)
            theta = u_el[1] + trapz_func(curv, x_eval, initial=0)
            defl = u_el[0] + trapz_func(theta, x_eval, initial=0)
            
            for k in range(len(x_eval)):
                results.append({
                    'x': self.nodes[i] + x_eval[k],
                    'shear': V_vals[k],
                    'moment': M_vals[k],
                    'deflection': defl[k]
                })

        Reactions = K_global @ U_global 
        # Note: No 'snap to zero' logic to preserve real moments at pins
        
        return pd.DataFrame(results), Reactions
