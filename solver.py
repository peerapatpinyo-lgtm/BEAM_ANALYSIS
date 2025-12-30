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
        
        # 1. Stiffness Matrix Assembly
        for i, L in enumerate(self.spans):
            k_el = (self.E * self.I / L**3) * np.array([
                [12, 6*L, -12, 6*L],
                [6*L, 4*L**2, -6*L, 2*L**2],
                [-12, -6*L, 12, -6*L],
                [6*L, 2*L**2, -6*L, 4*L**2]
            ])
            idx = [2*i, 2*i+1, 2*i+2, 2*i+3]
            for r in range(4):
                for c in range(4):
                    K_global[idx[r], idx[c]] += k_el[r, c]
            
            # FEM
            fem = np.zeros(4)
            span_loads = [l for l in self.loads if int(l['span_idx']) == i]
            for load in span_loads:
                if load['type'] == 'P':
                    a = load['x']; b = L - a; P = load['P']
                    fem[0] += P * b**2 * (3*a + b) / L**3
                    fem[1] += P * a * b**2 / L**2
                    fem[2] += P * a**2 * (a + 3*b) / L**3
                    fem[3] -= P * a**2 * b / L**2
                elif load['type'] == 'U':
                    w = load['w']
                    fem[0] += w * L / 2; fem[1] += w * L**2 / 12
                    fem[2] += w * L / 2; fem[3] -= w * L**2 / 12
            F_global[idx] -= fem 

        # 2. Boundary Conditions
        free_dofs = list(range(n_dof))
        for _, row in self.supports.iterrows():
            node_idx = int(row['id'])
            if 2*node_idx in free_dofs: free_dofs.remove(2*node_idx) # Fix Y
            if row['type'] == 'Fixed':
                if 2*node_idx+1 in free_dofs: free_dofs.remove(2*node_idx+1) # Fix Rotation

        # 3. Solve
        U_global = np.zeros(n_dof)
        if free_dofs:
            K_ff = K_global[np.ix_(free_dofs, free_dofs)]
            F_f = F_global[free_dofs]
            try:
                u_solved = np.linalg.solve(K_ff, F_f)
                U_global[free_dofs] = u_solved
            except np.linalg.LinAlgError:
                return pd.DataFrame(), np.zeros(n_dof) # Unstable

        # 4. Internal Forces & Deflection (Double Integration)
        results = []
        
        for i, L in enumerate(self.spans):
            idx = [2*i, 2*i+1, 2*i+2, 2*i+3]
            u_el = U_global[idx]
            
            # Local Boundary Conditions
            v_start = u_el[0]
            theta_start = u_el[1]
            
            # Member End Forces (Start)
            fem_local = np.zeros(4)
            span_loads = [l for l in self.loads if int(l['span_idx']) == i]
            for load in span_loads:
                if load['type'] == 'P':
                    a = load['x']; b = L - a; P = load['P']
                    fem_local[0] += P*b**2*(3*a+b)/L**3; fem_local[1] += P*a*b**2/L**2
                    fem_local[2] += P*a**2*(a+3*b)/L**3; fem_local[3] -= P*a**2*b/L**2
                elif load['type'] == 'U':
                    w = load['w']
                    fem_local[0] += w*L/2; fem_local[1] += w*L**2/12
                    fem_local[2] += w*L/2; fem_local[3] -= w*L**2/12
            
            k_el = (self.E * self.I / L**3) * np.array([
                [12, 6*L, -12, 6*L], [6*L, 4*L**2, -6*L, 2*L**2],
                [-12, -6*L, 12, -6*L], [6*L, 2*L**2, -6*L, 4*L**2]
            ])
            f_local = k_el @ u_el + fem_local
            V_start = f_local[0]
            M_start = f_local[1] # CCW+
            
            # High resolution for integration
            x_eval = np.linspace(0, L, 100)
            
            # Arrays to store values
            M_vals = []
            V_vals = []
            
            for x in x_eval:
                V_x = V_start
                M_x = -M_start + V_start * x
                
                for load in span_loads:
                    if load['type'] == 'P':
                        if x > load['x']:
                            V_x -= load['P']
                            M_x -= load['P'] * (x - load['x'])
                    elif load['type'] == 'U':
                        if x > 0:
                            w = load['w']
                            V_x -= w * x
                            M_x -= (w * x) * (x / 2)
                M_vals.append(M_x)
                V_vals.append(V_x)
            
            # Calculate Deflection via Integration
            # theta(x) = theta_start + integral(M/EI)
            # v(x) = v_start + integral(theta)
            M_vals = np.array(M_vals)
            curv = M_vals / (self.E * self.I)
            
            theta_vals = theta_start + integrate.cumulative_trapezoid(curv, x_eval, initial=0)
            deflection_vals = v_start + integrate.cumulative_trapezoid(theta_vals, x_eval, initial=0)
            
            # Store results
            for k in range(len(x_eval)):
                results.append({
                    'x': self.nodes[i] + x_eval[k],
                    'shear': V_vals[k],
                    'moment': M_vals[k],
                    'deflection': deflection_vals[k]
                })

        Reactions = K_global @ U_global 
        # Note: Reactions logic in display handles visual mapping
        
        return pd.DataFrame(results), Reactions
